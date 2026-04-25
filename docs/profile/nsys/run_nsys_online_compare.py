#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "compare"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "online"))

from compare_with_minisgl import (  # noqa: E402
    make_online_comparison_table,
    make_safe_synthetic_traces,
    run_online_benchmark_non_stream,
    wait_for_server,
)


def graph_batch_sizes() -> str:
    dense = list(range(1, 65)) + [80, 96, 112, 128]
    return ",".join(str(x) for x in dense)


def start_profiled_server(
    *,
    name: str,
    session: str,
    output_base: Path,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None,
    log_path: Path,
) -> subprocess.Popen[str]:
    launch_cmd = [
        "nsys",
        "launch",
        "--session-new",
        session,
        "--trace",
        "cuda,nvtx,osrt",
        "--cuda-graph-trace=graph",
        *cmd,
    ]
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        launch_cmd,
        cwd=str(cwd),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def nsys_start(session: str, output_base: Path) -> None:
    subprocess.run(
        [
            "nsys",
            "start",
            "--session",
            session,
            "--sample=none",
            "--cpuctxsw=none",
            "-o",
            str(output_base),
            "-f",
            "true",
        ],
        check=True,
    )


def nsys_stop(session: str) -> None:
    subprocess.run(["nsys", "stop", "--session", session], check=True)


def nsys_shutdown(session: str) -> None:
    subprocess.run(
        ["nsys", "shutdown", "--session", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


async def run_one_server(
    *,
    framework: str,
    session: str,
    output_dir: Path,
    server_cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None,
    port: int,
    traces: list[object],
    scales: list[float],
) -> pd.DataFrame:
    base_url = f"http://127.0.0.1:{port}"
    log_path = output_dir / f"{framework}_server.log"
    output_base = output_dir / f"{framework}_steady"

    nsys_shutdown(session)
    proc = start_profiled_server(
        name=framework,
        session=session,
        output_base=output_base,
        cmd=server_cmd,
        cwd=cwd,
        env=env,
        log_path=log_path,
    )
    try:
        wait_for_server(base_url, timeout_s=360)
        # Give the server one short idle interval after readiness so startup work is outside capture.
        time.sleep(1.0)
        nsys_start(session, output_base)
        frame = await run_online_benchmark_non_stream(
            framework=framework,
            base_url=base_url,
            traces=traces,
            scales=scales,
        )
        nsys_stop(session)
        return frame
    finally:
        stop_process(proc)
        nsys_shutdown(session)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "docs/profile/nsys/current_best"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mini-root", default=str(REPO_ROOT.parent / "mini-sglang"))
    parser.add_argument("--python", default="/root/miniconda3/bin/python")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    traces = make_safe_synthetic_traces(
        tokenizer,
        args.num_requests,
        seed=0,
        max_input_len=1024,
        min_output_len=16,
        max_output_len=256,
    )
    scales = [args.scale]

    common = [
        "--model-path",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--dtype",
        "bfloat16",
        "--max-running-requests",
        "256",
        "--max-prefill-length",
        "16384",
        "--page-size",
        "1",
        "--memory-ratio",
        "0.2",
        "--max-seq-len-override",
        "4096",
    ]

    sglang_cmd = [
        str(REPO_ROOT / "build" / "sglang_server"),
        *common,
        "--port",
        "1919",
        "--graph",
        "128",
        "--cuda-graph-capture-max-seq-len",
        "2048",
        "--cuda-graph-batch-sizes",
        graph_batch_sizes(),
    ]

    mini_env = os.environ.copy()
    mini_python = str(Path(args.mini_root) / "python")
    mini_env["PYTHONPATH"] = mini_python + (
        os.pathsep + mini_env["PYTHONPATH"] if mini_env.get("PYTHONPATH") else ""
    )
    mini_env["PATH"] = str(Path(args.python).resolve().parent) + os.pathsep + mini_env.get("PATH", "")
    mini_cmd = [
        args.python,
        "-m",
        "minisgl",
        *common,
        "--port",
        "1920",
    ]

    frames = []
    frames.append(
        await run_one_server(
            framework="sglang.cpp",
            session="sglang_cpp_nsys",
            output_dir=output_dir,
            server_cmd=sglang_cmd,
            cwd=REPO_ROOT,
            env=None,
            port=1919,
            traces=traces,
            scales=scales,
        )
    )
    frames.append(
        await run_one_server(
            framework="mini-sglang",
            session="mini_sglang_nsys",
            output_dir=output_dir,
            server_cmd=mini_cmd,
            cwd=Path(args.mini_root),
            env=mini_env,
            port=1920,
            traces=traces,
            scales=scales,
        )
    )

    results = pd.concat(frames, ignore_index=True)
    comparison = make_online_comparison_table(results)
    results.to_csv(output_dir / "online_results.csv", index=False)
    comparison.to_csv(output_dir / "online_comparison.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
