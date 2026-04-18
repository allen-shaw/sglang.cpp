# Benchmarks

This directory contains two benchmark tracks:

- `offline/`: C++ offline benchmarks built with Google Benchmark, focused on internal inference throughput.
- `online/`: Python end-to-end benchmarks that drive the HTTP server and print latency/throughput summaries.
- `compare/`: scripts that run `sglang.cpp` and `mini-sglang` side by side, export CSV summaries, and plot the gap.

## Offline benchmark

Build:

```bash
cmake --build build --target bench_offline -j4
```

Run:

```bash
./build/bench_offline \
  --model-path /path/to/model \
  --dtype bfloat16 \
  --batch-sizes 1,8,32,128 \
  --input-lens 128,512,1024 \
  --output-lens 128,512 \
  --benchmark_min_time=0.5
```

Use `--dummy-weight` to benchmark the scheduling/inference path without loading real weights.

## Online benchmark

Start the server first:

```bash
./build/sglang_server --model-path /path/to/model --host 127.0.0.1 --port 1919
```

Simple concurrency benchmark:

```bash
python benchmarks/online/bench_simple.py --host 127.0.0.1 --port 1919 --batch-sizes 8,16,64
```

Trace-style benchmark with a local Qwen trace:

```bash
python benchmarks/online/bench_qwen.py \
  --host 127.0.0.1 \
  --port 1919 \
  --trace-file /path/to/qwen_traceA_blksz_16.jsonl \
  --num-requests 1000 \
  --scales 0.4,0.5,0.6,0.7,0.8,1.6
```

If `--trace-file` is omitted or does not exist, `bench_qwen.py` falls back to synthetic traces.

## Compare with mini-sglang

Offline:

```bash
python /home/allen/workspace/mini-sglang/benchmark/offline/bench.py
./build/bench_offline --model-path /path/to/model --batch-sizes 256 --input-lens 1024 --output-lens 1024
```

Online:

```bash
python /home/allen/workspace/mini-sglang/benchmark/online/bench_qwen.py
python benchmarks/online/bench_qwen.py --host 127.0.0.1 --port 1919 --trace-file /path/to/qwen_traceA_blksz_16.jsonl
```

## Automated Comparison

The compare script runs both frameworks with the same model path and benchmark shape settings, then writes:

- raw framework outputs
- normalized CSV tables
- ratio CSV tables
- PNG plots for throughput/latency comparison

Offline-only example:

```bash
python benchmarks/compare/compare_with_minisgl.py \
  --mode offline \
  --model-path /path/to/model \
  --batch-sizes 1,8,32,128 \
  --input-lens 128,512,1024 \
  --output-lens 128,512 \
  --benchmark-min-time 1s
```

Online-only example:

```bash
python benchmarks/compare/compare_with_minisgl.py \
  --mode online \
  --model-path /path/to/model \
  --num-requests 256 \
  --scales 0.4,0.8,1.0
```

Run both:

```bash
python benchmarks/compare/compare_with_minisgl.py \
  --mode all \
  --model-path /path/to/model
```

Notes:

- By default, online comparison uses non-streaming `/generate` requests for robustness and compares `req/s`, `tok/s`, and E2E latency.
- Pass `--online-streaming` if you want to benchmark streaming behavior as well.
- Outputs are written under `benchmarks/results/<timestamp>/`.
