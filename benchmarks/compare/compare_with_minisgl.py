from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import aiohttp
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_BENCH_DIR = REPO_ROOT / "benchmarks" / "online"
if str(ONLINE_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(ONLINE_BENCH_DIR))

from bench_client import benchmark_trace, process_benchmark_results, read_qwen_trace, scale_traces  # noqa: E402


@dataclass(frozen=True)
class ServerHandle:
    framework: str
    process: subprocess.Popen[str]
    log_path: Path
    port: int


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare sglang.cpp and mini-sglang benchmark results")
    parser.add_argument("--mode", default="all", choices=["offline", "online", "all"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mini-root", default="/home/allen/workspace/mini-sglang")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sglang-bench", default=str(REPO_ROOT / "build_bench" / "bench_offline"))
    parser.add_argument("--sglang-server", default=str(REPO_ROOT / "build" / "sglang_server"))
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--dummy-weight", action="store_true")
    parser.add_argument("--max-running-requests", type=int, default=256)
    parser.add_argument("--max-prefill-length", type=int, default=16384)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--memory-ratio", type=float, default=0.2)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--max-seq-len-override", type=int, default=4096)
    parser.add_argument("--num-pages-override", type=int, default=0)
    parser.add_argument("--batch-sizes", default="1,8,32,128")
    parser.add_argument("--input-lens", default="128,512,1024")
    parser.add_argument("--output-lens", default="128,512")
    parser.add_argument("--benchmark-min-time", default="1s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="")

    parser.add_argument("--sglang-port", type=int, default=1919)
    parser.add_argument("--mini-port", type=int, default=1920)
    parser.add_argument("--trace-file", default="")
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--scales", default="0.4,0.8,1.0")
    parser.add_argument("--synthetic-max-input", type=int, default=1024)
    parser.add_argument("--synthetic-min-output", type=int, default=16)
    parser.add_argument("--synthetic-max-output", type=int, default=256)
    parser.add_argument("--online-streaming", action="store_true")
    parser.add_argument("--sglang-enable-graph", action="store_true")
    parser.add_argument("--sglang-graph-max-bs", type=int, default=0)
    parser.add_argument("--sglang-graph-batch-sizes", default="")
    parser.add_argument("--server-timeout", type=float, default=180.0)
    return parser


def parse_duration_seconds(value: str) -> float:
    if value.endswith("ms"):
        return float(value[:-2]) / 1000.0
    if value.endswith("s"):
        return float(value[:-1])
    return float(value)


def make_safe_prompt(tokenizer: Any, target_tokens: int, *, seed: int) -> tuple[str, int]:
    rng = random.Random(seed)
    prompt = "Benchmark prompt:"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    while len(token_ids) < target_tokens:
        extra = " " + " ".join(f"token{rng.randint(0, 1000)}" for _ in range(8))
        prompt += extra
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    return prompt, len(token_ids)


def make_safe_synthetic_traces(
    tokenizer: Any,
    n: int,
    *,
    seed: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
    avg_gap_s: float = 0.02,
) -> list[Any]:
    rng = random.Random(seed)
    current_time = 0.0
    traces: list[Any] = []
    for i in range(n):
        target_input_len = rng.randint(32, max_input_len)
        output_len = rng.randint(min_output_len, max_output_len)
        prompt, actual_input_len = make_safe_prompt(tokenizer, target_input_len, seed=seed + i)
        current_time += rng.expovariate(1.0 / avg_gap_s)
        traces.append(
            SimpleNamespace(
                timestamp=current_time,
                message=prompt,
                input_length=actual_input_len,
                output_length=output_len,
            )
        )
    return traces


def make_output_dir(arg_value: str) -> Path:
    if arg_value:
        path = Path(arg_value)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPO_ROOT / "benchmarks" / "results" / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if log_path is not None:
        log_path.write_text(result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else ""))
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def maybe_add_common_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    cmd.extend(
        [
            "--dtype",
            args.dtype,
            "--max-running-requests",
            str(args.max_running_requests),
            "--max-prefill-length",
            str(args.max_prefill_length),
            "--page-size",
            str(args.page_size),
            "--memory-ratio",
            str(args.memory_ratio),
        ]
    )
    if args.max_seq_len_override:
        cmd.extend(["--max-seq-len-override", str(args.max_seq_len_override)])
    if args.num_pages_override:
        cmd.extend(["--num-pages", str(args.num_pages_override)])
    if args.dummy_weight:
        cmd.append("--dummy-weight")
    return cmd


def run_sglang_offline(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    output_json = output_dir / "sglang_offline_raw.json"
    log_path = output_dir / "sglang_offline_cmd.log"
    cmd = [
        args.sglang_bench,
        "--model-path",
        args.model_path,
        "--batch-sizes",
        args.batch_sizes,
        "--input-lens",
        args.input_lens,
        "--output-lens",
        args.output_lens,
        "--warmup-tokens",
        str(args.warmup_tokens),
        f"--benchmark_min_time={args.benchmark_min_time}",
        "--benchmark_format=json",
        f"--benchmark_out={output_json}",
    ]
    cmd = maybe_add_common_args(cmd, args)
    run_command(cmd, cwd=REPO_ROOT, log_path=log_path)

    payload = json.loads(output_json.read_text())
    rows = []
    for bench in payload.get("benchmarks", []):
        if bench.get("run_type") != "iteration":
            continue
        rows.append(
            {
                "framework": "sglang.cpp",
                "batch_size": int(round(bench["batch_size"])),
                "input_len": int(round(bench["input_len"])),
                "output_len": int(round(bench["output_len"])),
                "prompt_tok/s": float(bench["prompt_tok/s"]),
                "gen_tok/s": float(bench["gen_tok/s"]),
                "req/s": float(bench["req/s"]),
                "duration_s": float(bench["real_time"]) / 1000.0,
                "iterations": int(bench["iterations"]),
            }
        )
    return pd.DataFrame(rows)


def run_minisgl_offline(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    output_json = output_dir / "mini_offline_raw.json"
    log_path = output_dir / "mini_offline_cmd.log"
    helper = REPO_ROOT / "benchmarks" / "compare" / "run_minisgl_offline.py"
    cmd = [
        args.python,
        str(helper),
        "--model-path",
        args.model_path,
        "--dtype",
        args.dtype,
        "--max-running-requests",
        str(args.max_running_requests),
        "--max-prefill-length",
        str(args.max_prefill_length),
        "--page-size",
        str(args.page_size),
        "--memory-ratio",
        str(args.memory_ratio),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--max-seq-len-override",
        str(args.max_seq_len_override),
        "--batch-sizes",
        args.batch_sizes,
        "--input-lens",
        args.input_lens,
        "--output-lens",
        args.output_lens,
        "--min-time-s",
        args.benchmark_min_time,
        "--seed",
        str(args.seed),
        "--output-json",
        str(output_json),
    ]
    if args.num_pages_override:
        cmd.extend(["--num-pages-override", str(args.num_pages_override)])
    if args.dummy_weight:
        cmd.append("--dummy-weight")

    env = os.environ.copy()
    mini_python = str(Path(args.mini_root) / "python")
    env["PYTHONPATH"] = mini_python + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    run_command(cmd, cwd=REPO_ROOT, env=env, log_path=log_path)

    payload = json.loads(output_json.read_text())
    return pd.DataFrame(payload["results"])


def plot_offline(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return

    df = df.copy()
    df["shape"] = df.apply(
        lambda row: f"bs={int(row.batch_size)}\nin={int(row.input_len)}\nout={int(row.output_len)}",
        axis=1,
    )
    for metric in ["gen_tok/s", "req/s", "prompt_tok/s"]:
        pivot = df.pivot(index="shape", columns="framework", values=metric)
        ax = pivot.plot(kind="bar", figsize=(14, 6), title=f"Offline {metric} comparison")
        ax.set_ylabel(metric)
        ax.set_xlabel("shape")
        ax.grid(axis="y", alpha=0.3)
        ax.figure.tight_layout()
        ax.figure.savefig(output_dir / f"offline_{metric.replace('/', '_per_')}.png", dpi=160)
        plt.close(ax.figure)


def make_offline_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["batch_size", "input_len", "output_len"],
        columns="framework",
        values=["gen_tok/s", "req/s", "prompt_tok/s"],
    )
    pivot.columns = [f"{metric}_{framework}" for metric, framework in pivot.columns]
    pivot = pivot.reset_index()
    if {"gen_tok/s_sglang.cpp", "gen_tok/s_mini-sglang"} <= set(pivot.columns):
        pivot["gen_tok_ratio_sglang_over_mini"] = (
            pivot["gen_tok/s_sglang.cpp"] / pivot["gen_tok/s_mini-sglang"]
        )
    if {"req/s_sglang.cpp", "req/s_mini-sglang"} <= set(pivot.columns):
        pivot["req_ratio_sglang_over_mini"] = pivot["req/s_sglang.cpp"] / pivot["req/s_mini-sglang"]
    if {"prompt_tok/s_sglang.cpp", "prompt_tok/s_mini-sglang"} <= set(pivot.columns):
        pivot["prompt_tok_ratio_sglang_over_mini"] = (
            pivot["prompt_tok/s_sglang.cpp"] / pivot["prompt_tok/s_mini-sglang"]
        )
    return pivot


def plot_offline_ratio(df: pd.DataFrame, output_dir: Path) -> None:
    ratio_columns = [
        column
        for column in df.columns
        if column.endswith("_ratio_sglang_over_mini")
    ]
    if not ratio_columns:
        return
    ratio_df = df.copy()
    ratio_df["shape"] = ratio_df.apply(
        lambda row: f"bs={int(row.batch_size)}\nin={int(row.input_len)}\nout={int(row.output_len)}",
        axis=1,
    )
    plot_df = ratio_df.set_index("shape")[ratio_columns]
    ax = plot_df.plot(kind="bar", figsize=(14, 6), title="Offline ratio: sglang.cpp / mini-sglang")
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("ratio")
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "offline_ratio.png", dpi=160)
    plt.close(ax.figure)


def wait_for_server(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = "server not ready"
    while time.time() < deadline:
        try:
            with urlopen(f"{base_url}/v1/models", timeout=2.0) as response:
                if response.status == 200:
                    return
        except URLError as error:
            last_error = str(error)
        except Exception as error:  # pragma: no cover - best-effort polling
            last_error = str(error)
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for server {base_url}: {last_error}")


def start_sglang_server(args: argparse.Namespace, output_dir: Path) -> ServerHandle:
    log_path = output_dir / "sglang_server.log"
    cmd = [
        args.sglang_server,
        "--model-path",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.sglang_port),
    ]
    cmd = maybe_add_common_args(cmd, args)
    if args.sglang_enable_graph:
        graph_max_bs = args.sglang_graph_max_bs or args.max_running_requests
        cmd.extend(["--graph", str(graph_max_bs)])
        if args.sglang_graph_batch_sizes:
            cmd.extend(["--cuda-graph-batch-sizes", args.sglang_graph_batch_sizes])
    log_file = log_path.open("w")
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ServerHandle("sglang.cpp", process, log_path, args.sglang_port)


def start_minisgl_server(args: argparse.Namespace, output_dir: Path) -> ServerHandle:
    log_path = output_dir / "mini_server.log"
    cmd = [
        args.python,
        "-m",
        "minisgl",
        "--model-path",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.mini_port),
        "--dtype",
        args.dtype,
        "--max-running-requests",
        str(args.max_running_requests),
        "--max-prefill-length",
        str(args.max_prefill_length),
        "--page-size",
        str(args.page_size),
        "--memory-ratio",
        str(args.memory_ratio),
    ]
    if args.max_seq_len_override:
        cmd.extend(["--max-seq-len-override", str(args.max_seq_len_override)])
    if args.num_pages_override:
        cmd.extend(["--num-pages", str(args.num_pages_override)])
    if args.dummy_weight:
        cmd.append("--dummy-weight")

    env = os.environ.copy()
    mini_python = str(Path(args.mini_root) / "python")
    env["PYTHONPATH"] = mini_python + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_file = log_path.open("w")
    process = subprocess.Popen(
        cmd,
        cwd=str(Path(args.mini_root)),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ServerHandle("mini-sglang", process, log_path, args.mini_port)


def stop_server(handle: ServerHandle) -> None:
    if handle.process.poll() is not None:
        return
    handle.process.terminate()
    try:
        handle.process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        handle.process.kill()
        handle.process.wait(timeout=5)


async def run_online_benchmark_once(
    *,
    framework: str,
    base_url: str,
    traces: list[Any],
    scales: list[float],
) -> pd.DataFrame:
    rows = []
    for scale in scales:
        scaled = scale_traces(traces, scale)
        results = await benchmark_trace(base_url, scaled)
        summary = process_benchmark_results(results)
        summary["scale"] = scale
        summary["framework"] = framework
        rows.append(summary)
    return pd.DataFrame(rows)


async def run_non_stream_request(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    output_len: int,
) -> tuple[float, float, int]:
    payload = {
        "prompt": prompt,
        "max_tokens": output_len,
        "ignore_eos": True,
        "stream": False,
    }
    t0 = time.perf_counter()
    async with session.post(f"{base_url}/generate", json=payload) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("text/event-stream"):
            while True:
                line = await response.content.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text.startswith("data:"):
                    continue
                data = text[5:].strip()
                if data == "[DONE]":
                    break
        else:
            await response.json()
    t1 = time.perf_counter()
    return t0, t1, output_len


async def run_online_benchmark_non_stream(
    *,
    framework: str,
    base_url: str,
    traces: list[Any],
    scales: list[float],
) -> pd.DataFrame:
    rows = []
    timeout = aiohttp.ClientTimeout(total=None)

    for scale in scales:
        scaled = scale_traces(traces, scale)
        start = time.perf_counter()
        offset = min(trace.timestamp for trace in scaled) if scaled else 0.0

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async def run_one(trace: Any) -> tuple[float, float, int]:
                target = start + trace.timestamp - offset
                await asyncio.sleep(max(0.0, target - time.perf_counter()))
                return await run_non_stream_request(
                    session,
                    base_url,
                    trace.message,
                    trace.output_length,
                )

            raw_results = await asyncio.gather(*(run_one(trace) for trace in scaled))

        start_times = [item[0] for item in raw_results]
        end_times = [item[1] for item in raw_results]
        e2e_times = sorted(item[1] - item[0] for item in raw_results)
        total_output_tokens = sum(item[2] for item in raw_results)
        duration = max(end_times) - min(start_times) if raw_results else 0.0
        rows.append(
            {
                "framework": framework,
                "scale": scale,
                "num_requests": float(len(raw_results)),
                "total_output_tokens": float(total_output_tokens),
                "duration_s": duration,
                "req_per_s": len(raw_results) / duration if duration > 0 else 0.0,
                "tok_per_s": total_output_tokens / duration if duration > 0 else 0.0,
                "avg_e2e_s": sum(e2e_times) / len(e2e_times) if e2e_times else 0.0,
                "p50_e2e_s": e2e_times[len(e2e_times) // 2] if e2e_times else 0.0,
                "p90_e2e_s": e2e_times[min(len(e2e_times) - 1, int(len(e2e_times) * 0.9))]
                if e2e_times
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def make_online_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="scale",
        columns="framework",
        values=[
            column
            for column in ["req_per_s", "tok_per_s", "avg_ttft_ms", "avg_e2e_s", "p90_e2e_s"]
            if column in df.columns
        ],
    )
    pivot.columns = [f"{metric}_{framework}" for metric, framework in pivot.columns]
    pivot = pivot.reset_index()
    if {"req_per_s_sglang.cpp", "req_per_s_mini-sglang"} <= set(pivot.columns):
        pivot["req_ratio_sglang_over_mini"] = (
            pivot["req_per_s_sglang.cpp"] / pivot["req_per_s_mini-sglang"]
        )
    if {"tok_per_s_sglang.cpp", "tok_per_s_mini-sglang"} <= set(pivot.columns):
        pivot["tok_ratio_sglang_over_mini"] = (
            pivot["tok_per_s_sglang.cpp"] / pivot["tok_per_s_mini-sglang"]
        )
    if {"avg_ttft_ms_sglang.cpp", "avg_ttft_ms_mini-sglang"} <= set(pivot.columns):
        pivot["avg_ttft_ratio_sglang_over_mini"] = (
            pivot["avg_ttft_ms_sglang.cpp"] / pivot["avg_ttft_ms_mini-sglang"]
        )
    if {"avg_e2e_s_sglang.cpp", "avg_e2e_s_mini-sglang"} <= set(pivot.columns):
        pivot["avg_e2e_ratio_sglang_over_mini"] = (
            pivot["avg_e2e_s_sglang.cpp"] / pivot["avg_e2e_s_mini-sglang"]
        )
    if {"p90_e2e_s_sglang.cpp", "p90_e2e_s_mini-sglang"} <= set(pivot.columns):
        pivot["p90_e2e_ratio_sglang_over_mini"] = (
            pivot["p90_e2e_s_sglang.cpp"] / pivot["p90_e2e_s_mini-sglang"]
        )
    return pivot


def load_or_make_traces(args: argparse.Namespace) -> list[Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    trace_path = Path(args.trace_file) if args.trace_file else None
    if trace_path and trace_path.exists():
        return read_qwen_trace(str(trace_path), tokenizer, n=args.num_requests)
    return make_safe_synthetic_traces(
        tokenizer,
        args.num_requests,
        seed=args.seed,
        max_input_len=args.synthetic_max_input,
        min_output_len=args.synthetic_min_output,
        max_output_len=args.synthetic_max_output,
    )


def run_online_compare(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    traces = load_or_make_traces(args)
    scales = parse_csv_floats(args.scales)

    frames = []
    for starter in [start_sglang_server, start_minisgl_server]:
        handle = starter(args, output_dir)
        base_url = f"http://127.0.0.1:{handle.port}"
        try:
            wait_for_server(base_url, args.server_timeout)
            frame = asyncio.run(
                (
                    run_online_benchmark_once(
                        framework=handle.framework,
                        base_url=base_url,
                        traces=traces,
                        scales=scales,
                    )
                    if args.online_streaming
                    else run_online_benchmark_non_stream(
                        framework=handle.framework,
                        base_url=base_url,
                        traces=traces,
                        scales=scales,
                    )
                )
            )
            frames.append(frame)
        finally:
            stop_server(handle)
    return pd.concat(frames, ignore_index=True)


def plot_online(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    metrics = [
        metric
        for metric in ["req_per_s", "tok_per_s", "avg_ttft_ms", "avg_e2e_s", "p90_e2e_s"]
        if metric in df.columns
    ]
    for metric in metrics:
        pivot = df.pivot(index="scale", columns="framework", values=metric)
        ax = pivot.plot(marker="o", figsize=(10, 5), title=f"Online {metric} by scale")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.figure.tight_layout()
        ax.figure.savefig(output_dir / f"online_{metric}.png", dpi=160)
        plt.close(ax.figure)


def plot_online_ratio(df: pd.DataFrame, output_dir: Path) -> None:
    ratio_columns = [
        column
        for column in df.columns
        if column.endswith("_ratio_sglang_over_mini")
    ]
    if not ratio_columns:
        return
    plot_df = df.set_index("scale")[ratio_columns]
    ax = plot_df.plot(marker="o", figsize=(10, 5), title="Online ratio: sglang.cpp / mini-sglang")
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("ratio")
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "online_ratio.png", dpi=160)
    plt.close(ax.figure)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = make_output_dir(args.output_dir)
    print(f"Writing benchmark comparison outputs to {output_dir}")

    summary: dict[str, str] = {}

    if args.mode in {"offline", "all"}:
        offline_frames = [
            run_sglang_offline(args, output_dir),
            run_minisgl_offline(args, output_dir),
        ]
        offline_df = pd.concat(offline_frames, ignore_index=True)
        offline_csv = output_dir / "offline_results.csv"
        offline_df.to_csv(offline_csv, index=False)
        comparison_df = make_offline_comparison_table(offline_df)
        comparison_csv = output_dir / "offline_comparison.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        plot_offline(offline_df, output_dir)
        plot_offline_ratio(comparison_df, output_dir)
        summary["offline_results"] = str(offline_csv)
        summary["offline_comparison"] = str(comparison_csv)

    if args.mode in {"online", "all"}:
        online_df = run_online_compare(args, output_dir)
        online_csv = output_dir / "online_results.csv"
        online_df.to_csv(online_csv, index=False)
        online_comparison_df = make_online_comparison_table(online_df)
        online_comparison_csv = output_dir / "online_comparison.csv"
        online_comparison_df.to_csv(online_comparison_csv, index=False)
        plot_online(online_df, output_dir)
        plot_online_ratio(online_comparison_df, output_dir)
        summary["online_results"] = str(online_csv)
        summary["online_comparison"] = str(online_comparison_csv)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
