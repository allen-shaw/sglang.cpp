from __future__ import annotations

import argparse
import asyncio
import random

import aiohttp
from transformers import AutoTokenizer

from bench_client import (
    benchmark_one,
    benchmark_one_batch,
    generate_prompt,
    get_model_name,
    process_benchmark_results,
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple online benchmark for sglang.cpp")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--batch-sizes", default="64")
    parser.add_argument("--max-input", type=int, default=8192)
    parser.add_argument("--min-output", type=int, default=16)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)

    base_url = f"http://{args.host}:{args.port}"
    batch_sizes = parse_csv_ints(args.batch_sizes)
    max_bs = max(batch_sizes)

    model_name = await get_model_name(base_url)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loaded tokenizer from {model_name}")

    warmup_prompt = generate_prompt(tokenizer, 100, seed=args.seed)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        await benchmark_one(
            session=session,
            base_url=base_url,
            prompt=warmup_prompt,
            output_length=2,
        )

    prompts = [
        generate_prompt(tokenizer, random.randint(1, args.max_input), seed=args.seed + i)
        for i in range(max_bs)
    ]
    output_lengths = [random.randint(args.min_output, args.max_output) for _ in range(max_bs)]

    for batch_size in batch_sizes:
        print(f"Running online simple benchmark with batch_size={batch_size}")
        results = await benchmark_one_batch(
            base_url,
            prompts[:batch_size],
            output_lengths[:batch_size],
        )
        process_benchmark_results(results)


if __name__ == "__main__":
    asyncio.run(main())
