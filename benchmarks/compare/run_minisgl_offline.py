from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_duration_seconds(value: str) -> float:
    if value.endswith("ms"):
        return float(value[:-2]) / 1000.0
    if value.endswith("s"):
        return float(value[:-1])
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Structured offline benchmark runner for mini-sglang")
    parser.add_argument("--model-path", required=True)
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
    parser.add_argument("--min-time-s", default="1s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    return parser


def parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[value]


def make_prompt_token_ids(
    batch_size: int,
    input_len: int,
    output_len: int,
    *,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed + batch_size * 17 + input_len * 31 + output_len * 13)
    return [[rng.randint(0, 10000) for _ in range(input_len)] for _ in range(batch_size)]


def make_sampling_params(batch_size: int, output_len: int) -> list[SamplingParams]:
    return [
        SamplingParams(
            temperature=0.6,
            top_p=1.0,
            top_k=-1,
            ignore_eos=True,
            max_tokens=output_len,
        )
        for _ in range(batch_size)
    ]


def output_token_count(result: object) -> int:
    if isinstance(result, dict) and "token_ids" in result:
        return len(result["token_ids"])  # type: ignore[arg-type]
    if isinstance(result, list):
        return len(result)
    if hasattr(result, "__len__"):
        return len(result)  # type: ignore[arg-type]
    raise TypeError(f"Unsupported mini-sglang output type: {type(result)!r}")


def main() -> None:
    args = build_parser().parse_args()

    llm = LLM(
        model_path=args.model_path,
        dtype=parse_dtype(args.dtype),
        max_running_req=args.max_running_requests,
        page_size=args.page_size,
        memory_ratio=args.memory_ratio,
        max_extend_tokens=args.max_prefill_length,
        max_seq_len_override=args.max_seq_len_override,
        num_page_override=(args.num_pages_override or None),
        use_dummy_weight=args.dummy_weight,
    )

    llm.generate(["Benchmark warmup"], SamplingParams(temperature=0.1, max_tokens=args.warmup_tokens))

    rows: list[dict[str, float | int | str]] = []
    batch_sizes = parse_csv_ints(args.batch_sizes)
    input_lens = parse_csv_ints(args.input_lens)
    output_lens = parse_csv_ints(args.output_lens)
    min_time_s = parse_duration_seconds(args.min_time_s)

    for batch_size in batch_sizes:
        for input_len in input_lens:
            for output_len in output_lens:
                prompt_token_ids = make_prompt_token_ids(
                    batch_size,
                    input_len,
                    output_len,
                    seed=args.seed,
                )
                sampling_params = make_sampling_params(batch_size, output_len)

                elapsed_s = 0.0
                iterations = 0
                generated_tokens = 0

                while elapsed_s < min_time_s or iterations == 0:
                    t0 = time.perf_counter()
                    outputs = llm.generate(prompt_token_ids, sampling_params)
                    elapsed_s += time.perf_counter() - t0
                    iterations += 1
                    generated_tokens += sum(output_token_count(output) for output in outputs)

                prompt_tokens = iterations * batch_size * input_len
                total_requests = iterations * batch_size
                rows.append(
                    {
                        "framework": "mini-sglang",
                        "batch_size": batch_size,
                        "input_len": input_len,
                        "output_len": output_len,
                        "iterations": iterations,
                        "duration_s": elapsed_s,
                        "prompt_tok/s": prompt_tokens / elapsed_s,
                        "gen_tok/s": generated_tokens / elapsed_s,
                        "req/s": total_requests / elapsed_s,
                    }
                )

    payload = {
        "framework": "mini-sglang",
        "config": vars(args),
        "results": rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
