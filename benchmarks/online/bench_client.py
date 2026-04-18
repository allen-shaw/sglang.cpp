from __future__ import annotations

import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import aiohttp


@dataclass(frozen=True)
class BenchmarkTrace:
    timestamp: float
    message: str
    output_length: int
    input_length: int | None = None


@dataclass(frozen=True)
class RawResult:
    input_len: int | None
    output_len: int
    message: str
    tics: list[float]


def generate_prompt(tokenizer: Any, n: int, seed: int | None = None) -> str:
    rng = random.Random(seed)
    vocab_size = max(1, tokenizer.vocab_size // 2)
    token_ids = [rng.randint(0, vocab_size) for _ in range(n)]

    for _ in range(64):
        prompt = tokenizer.decode(token_ids)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) == n:
            return prompt
        if len(token_ids) < n:
            token_ids.extend(rng.randint(0, vocab_size) for _ in range(n - len(token_ids)))
        else:
            token_ids = token_ids[:n]

    raise ValueError("failed to synthesize prompt with desired token length")


async def get_model_name(base_url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/v1/models") as response:
            response.raise_for_status()
            payload = await response.json()
            data = payload.get("data", [])
            if not data:
                raise ValueError("server returned no models")
            return data[0]["id"]


async def benchmark_one(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    output_length: int,
    *,
    input_length: int | None = None,
) -> RawResult:
    payload = {
        "prompt": prompt,
        "max_tokens": output_length,
        "ignore_eos": True,
        "stream": True,
    }

    tics = [time.perf_counter()]
    async with session.post(f"{base_url}/generate", json=payload) as response:
        response.raise_for_status()
        while True:
            line = await response.content.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text.startswith("data:"):
                continue
            data = text[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                break
            tics.append(time.perf_counter())

    return RawResult(
        input_len=input_length,
        output_len=output_length,
        message=prompt,
        tics=tics,
    )


async def benchmark_one_batch(
    base_url: str,
    prompts: list[str],
    output_lengths: list[int] | int,
    *,
    input_lengths: list[int | None] | None = None,
) -> list[RawResult]:
    if isinstance(output_lengths, int):
        output_lengths = [output_lengths] * len(prompts)
    if input_lengths is None:
        input_lengths = [None] * len(prompts)

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            benchmark_one(
                session,
                base_url,
                prompt,
                output_len,
                input_length=input_len,
            )
            for prompt, output_len, input_len in zip(
                prompts, output_lengths, input_lengths, strict=True
            )
        ]
        return await asyncio.gather(*tasks)


async def benchmark_trace(
    base_url: str,
    traces: list[BenchmarkTrace],
) -> list[RawResult]:
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        start = time.perf_counter()
        offset = min(trace.timestamp for trace in traces)

        async def run_one(trace: BenchmarkTrace) -> RawResult:
            target = start + trace.timestamp - offset
            await asyncio.sleep(max(0.0, target - time.perf_counter()))
            return await benchmark_one(
                session,
                base_url,
                trace.message,
                trace.output_length,
                input_length=trace.input_length,
            )

        return await asyncio.gather(*(run_one(trace) for trace in traces))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(len(values) * pct))
    return values[idx]


def process_benchmark_results(results: list[RawResult]) -> dict[str, float]:
    first_times = []
    per_token_times = []
    e2e_times = []

    min_time = min(result.tics[0] for result in results)
    max_time = max(result.tics[-1] for result in results)
    duration = max_time - min_time

    for result in results:
        if len(result.tics) >= 2:
            deltas = [result.tics[i + 1] - result.tics[i] for i in range(len(result.tics) - 1)]
            first_times.append(deltas[0])
            per_token_times.extend(deltas[1:])
        e2e_times.append(result.tics[-1] - result.tics[0])

    first_times.sort()
    per_token_times.sort()
    e2e_times.sort()

    total_output_tokens = sum(result.output_len for result in results)
    total_requests = len(results)
    summary = {
        "num_requests": float(total_requests),
        "total_output_tokens": float(total_output_tokens),
        "duration_s": duration,
        "req_per_s": total_requests / duration if duration > 0 else 0.0,
        "tok_per_s": total_output_tokens / duration if duration > 0 else 0.0,
        "avg_ttft_ms": statistics.fmean(first_times) * 1000 if first_times else 0.0,
        "p50_ttft_ms": _percentile(first_times, 0.5) * 1000 if first_times else 0.0,
        "p90_ttft_ms": _percentile(first_times, 0.9) * 1000 if first_times else 0.0,
        "avg_tpot_ms": statistics.fmean(per_token_times) * 1000 if per_token_times else 0.0,
        "p50_tpot_ms": _percentile(per_token_times, 0.5) * 1000 if per_token_times else 0.0,
        "p90_tpot_ms": _percentile(per_token_times, 0.9) * 1000 if per_token_times else 0.0,
        "avg_e2e_s": statistics.fmean(e2e_times) if e2e_times else 0.0,
        "p50_e2e_s": _percentile(e2e_times, 0.5) if e2e_times else 0.0,
        "p90_e2e_s": _percentile(e2e_times, 0.9) if e2e_times else 0.0,
    }

    print(json.dumps(summary, indent=2))
    return summary


def read_qwen_trace(trace_path: str, tokenizer: Any, n: int | None = None) -> list[BenchmarkTrace]:
    lines = Path(trace_path).read_text().splitlines()
    if n is not None:
        lines = lines[:n]

    rows = [json.loads(line) for line in lines]
    max_input_length = max(row["input_length"] for row in rows)
    base_prompt = generate_prompt(tokenizer, max_input_length, seed=0)
    base_ids = tokenizer.encode(base_prompt, add_special_tokens=False)

    traces = []
    for row in rows:
        prompt = tokenizer.decode(base_ids[: row["input_length"]])
        traces.append(
            BenchmarkTrace(
                timestamp=float(row["timestamp"]),
                message=prompt,
                input_length=int(row["input_length"]),
                output_length=int(row["output_length"]),
            )
        )
    return traces


def make_synthetic_traces(
    tokenizer: Any,
    n: int,
    *,
    seed: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
    avg_gap_s: float = 0.02,
) -> list[BenchmarkTrace]:
    rng = random.Random(seed)
    current_time = 0.0
    traces = []
    for i in range(n):
        input_len = rng.randint(32, max_input_len)
        output_len = rng.randint(min_output_len, max_output_len)
        prompt = generate_prompt(tokenizer, input_len, seed=seed + i)
        current_time += rng.expovariate(1.0 / avg_gap_s)
        traces.append(
            BenchmarkTrace(
                timestamp=current_time,
                message=prompt,
                input_length=input_len,
                output_length=output_len,
            )
        )
    return traces


def scale_traces(traces: Iterable[BenchmarkTrace], scale: float) -> list[BenchmarkTrace]:
    traces = list(traces)
    if not traces:
        return []
    min_timestamp = min(trace.timestamp for trace in traces)
    return [
        BenchmarkTrace(
            timestamp=(trace.timestamp - min_timestamp) * scale,
            message=trace.message,
            input_length=trace.input_length,
            output_length=trace.output_length,
        )
        for trace in traces
    ]
