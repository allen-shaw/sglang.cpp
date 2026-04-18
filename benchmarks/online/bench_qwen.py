from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from transformers import AutoTokenizer

from bench_client import (
    benchmark_trace,
    get_model_name,
    make_synthetic_traces,
    process_benchmark_results,
    read_qwen_trace,
    scale_traces,
)


def parse_csv_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen trace-style online benchmark for sglang.cpp")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--trace-file", default="")
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--scales", default="0.4,0.5,0.6,0.7,0.8,1.6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-max-input", type=int, default=1024)
    parser.add_argument("--synthetic-min-output", type=int, default=16)
    parser.add_argument("--synthetic-max-output", type=int, default=256)
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    base_url = f"http://{args.host}:{args.port}"
    scales = parse_csv_floats(args.scales)

    model_name = await get_model_name(base_url)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loaded tokenizer from {model_name}")

    trace_path = Path(args.trace_file) if args.trace_file else None
    if trace_path and trace_path.exists():
        traces = read_qwen_trace(str(trace_path), tokenizer, n=args.num_requests)
        print(f"Loaded {len(traces)} traces from {trace_path}")
    else:
        traces = make_synthetic_traces(
            tokenizer,
            args.num_requests,
            seed=args.seed,
            max_input_len=args.synthetic_max_input,
            min_output_len=args.synthetic_min_output,
            max_output_len=args.synthetic_max_output,
        )
        print(f"Using {len(traces)} synthetic traces")

    for scale in scales:
        print(f"Running trace benchmark at scale={scale}")
        scaled = scale_traces(traces, scale)
        results = await benchmark_trace(base_url, scaled)
        process_benchmark_results(results)


if __name__ == "__main__":
    asyncio.run(main())
