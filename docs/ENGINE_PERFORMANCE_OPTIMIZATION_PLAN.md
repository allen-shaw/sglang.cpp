# sglang.cpp Engine Performance Optimization Plan

Date: 2026-04-25

## Summary

This plan follows the current profiling evidence in `docs/profile/report.md` and the aligned `nsys` artifacts in `docs/profile/nsys/current_best/`.

The current best `sglang.cpp` configuration is stable but still below `mini-sglang`:

- `tok_ratio_sglang_over_mini`: `0.9487`
- `req_ratio_sglang_over_mini`: `0.9487`
- `avg_e2e_ratio_sglang_over_mini`: `1.1527`
- `p90_e2e_ratio_sglang_over_mini`: `1.1415`

The remaining bottleneck is in Engine execution rather than Scheduler scheduling. The optimization order is:

1. Reduce Engine host/API overhead around metadata staging, `cudaMemcpyAsync`, stream synchronization, and CUDA graph replay.
2. Replace slower Engine elementwise CUDA kernels for `silu_and_mul`, RMSNorm/qk-norm/fused-add-RMSNorm, and rotary.
3. Remove residual ATen indexing/copy kernels and reduce non-graph launch overhead.

The acceptance target remains the online `/generate` comparison against `mini-sglang`: `sglang.cpp` must exceed mini on `tok/s` and `req/s`, and must not regress `avg_e2e` or `p90_e2e`.

## Current Evidence

Aligned `nsys` profiles were collected with the same online request shape:

- Artifacts: `docs/profile/nsys/current_best/`
- `sglang.cpp`: `--graph 128 --cuda-graph-capture-max-seq-len 2048 --cuda-graph-bs 1,2,...,64,80,96,112,128`
- Workload: online `/generate`, scale `1.0`, 128 synthetic non-streaming requests

End-to-end result under `nsys`:

| metric | mini-sglang | sglang.cpp | ratio |
|---|---:|---:|---:|
| tok/s | 3787.02 | 3592.66 | 0.9487 |
| req/s | 26.59 | 25.23 | 0.9487 |
| avg E2E | 1.696 s | 1.955 s | 1.1527 |
| p90 E2E | 2.730 s | 3.116 s | 1.1415 |

CUDA API summary:

| item | sglang.cpp | mini-sglang | finding |
|---|---:|---:|---|
| total CUDA API time | 4388 ms | 3483 ms | sglang.cpp +905 ms |
| `cudaEventSynchronize` | 3541 ms / 3867 calls | 3065 ms / 1222 calls | sglang.cpp waits more frequently |
| `cudaMemcpyAsync` | 312 ms / 18926 calls | 61 ms / 10146 calls | sglang.cpp has much higher metadata/copy API overhead |
| `cudaStreamSynchronize` | 210 ms / 613 calls | not a top item | sglang.cpp has extra stream sync cost |
| kernel launch APIs | 205.7 ms / 56828 calls | 255.4 ms / 43555 calls | sglang.cpp launches about 30% more non-graph kernels |
| `cudaGraphLaunch` | 71 ms / 483 calls | 36 ms / 512 calls | sglang.cpp graph replay API is about 2x average cost |

GPU kernel summary:

| item | sglang.cpp | mini-sglang | finding |
|---|---:|---:|---|
| total GPU kernel time | 1229 ms | 1178 ms | sglang.cpp GPU kernels are about 4.3% slower overall |
| prefill attention | 167.3 ms / 2660 | 165.1 ms / 2744 | attention is not the main gap |
| top BF16 GEMM | 320.5 ms / 4032 | 308.3 ms / 3948 | similar, small sglang.cpp overhead |
| RMSNorm family | 47.3 ms / 10735 | 33.0 ms / 11187 | mini uses faster/fused norm kernels |
| rotary | 25.3 ms / 2660 | 14.6 ms / 2772 | sglang.cpp rotary is about 1.7x slower |
| activation `silu_and_mul` | 42.8 ms / 2660 | 14.4 ms / 2772 | sglang.cpp activation is about 3x slower |
| sglang.cpp `index_copy` | 21.4 ms / 5320 | n/a | PyTorch index/copy path remains visible |

## Optimization Phases

### Phase 1: Reduce Engine Host/API Overhead

Goal: reduce per-step CUDA API work before changing model math kernels.

Implementation targets:

- Inspect `GraphRunner`, `FlashInferBackend::prepare_for_replay`, and decode/prefill metadata construction paths.
- Reduce per-step `cudaMemcpyAsync` call count and metadata host staging.
- Replace the current single pinned workspace plus event wait protection with a ring-buffered or double-buffered pinned workspace.
- Keep overlap correctness intact: all scheduler-produced tensors consumed on the engine stream must remain protected with allocator `recordStream` or explicit CUDA event ordering.
- Avoid adding global stream synchronizations. Any new synchronization must be local, ordered, and justified by data dependency.

Expected profile movement:

- `cudaMemcpyAsync` API total and call count decrease substantially from `312 ms / 18926 calls`.
- `cudaStreamSynchronize` should stop appearing as a major API item.
- `cudaGraphLaunch` average cost should move closer to mini-sglang.

### Phase 2: Align Faster Elementwise Kernel Paths

Goal: remove the GPU-side gap in model-path non-GEMM kernels.

Implementation targets:

- Replace or port a faster fused `silu_and_mul` path. This is the highest-value kernel target because `sglang.cpp` currently spends `42.8 ms` versus mini's `14.4 ms`.
- Replace RMSNorm/qk-norm/fused-add-RMSNorm with faster fused kernels where shapes and memory layout match the model path.
- Optimize rotary by matching mini-sglang's faster kernel signature, parameter layout, and launch path where applicable.
- Do not prioritize attention or major GEMM first; current aligned `nsys` data shows those are already close.

Expected profile movement:

- `silu_and_mul` approaches mini-sglang's `14.4 ms` total time.
- RMSNorm family approaches mini-sglang's `33.0 ms`.
- Rotary approaches mini-sglang's `14.6 ms`.
- Total GPU kernel time moves from `1229 ms` to at or below mini-sglang's `1178 ms`.

### Phase 3: Reduce ATen Indexing/Copy and Non-Graph Launches

Goal: remove residual general-purpose tensor operations from the decode hot path.

Implementation targets:

- Identify hot `at::native::index_copy_kernel`, elementwise copy/add, and reduce kernels in current `nsys` output.
- Replace hot ATen indexing/copy calls with custom CUDA kernels where the access pattern is known and stable.
- Fold simple metadata/token/cache update work into existing kernels when doing so reduces launches without obscuring correctness.
- Expand CUDA graph coverage or fix graph fallback cases that still produce eager launches in steady-state decode.

Expected profile movement:

- ATen indexing/copy kernel time decreases from the current visible `index_copy` cost of about `21.4 ms`.
- Non-graph launch count decreases from current `sglang.cpp` levels.
- Graph replay steady-state becomes closer to mini-sglang in both API count and latency.

## Benchmark and Profile Protocol

After every optimization phase, save artifacts under `docs/profile/round-N/`.

Build and correctness:

```bash
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Key correctness tests:

- `TestEngineE2E`
- `TestSchedulerE2E`
- `TestHttpE2E`
- `TestLLM`

Performance comparison:

```bash
/root/miniconda3/bin/python benchmarks/compare/compare_with_minisgl.py --mode online
```

Use the fixed model path:

```text
/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

Run online comparison at scale `0.4`, `0.8`, and `1.0`.

Final acceptance requires all three online scales to satisfy:

- `tok_ratio_sglang_over_mini > 1.0`
- `req_ratio_sglang_over_mini > 1.0`
- `avg_e2e_ratio_sglang_over_mini <= 1.0`
- `p90_e2e_ratio_sglang_over_mini <= 1.0`

Profile comparison:

- Re-run aligned `nsys` for `sglang.cpp` and mini-sglang after each phase.
- Export `cuda_api_sum`, `cuda_gpu_kern_sum`, `cuda_kern_exec_sum`, and `cuda_gpu_mem_time_sum`.
- Update `docs/profile/report.md` with the phase result, profile deltas, and remaining bottlenecks.

## Assumptions and Constraints

- Keep the current overlap correctness fixes. Do not remove scheduler tensor `recordStream` protection or FlashInfer workspace event ordering without replacing them with an equivalent correctness mechanism.
- Use `mini-sglang` as the performance reference, not as a correctness oracle for changing model outputs.
- Do not pursue unprofiled refactors. Each optimization must target a measured bottleneck from `docs/profile/report.md` or a new round-specific profile.
- The current full `ctest` suite has known unrelated low-level failures. Treat regressions in the key E2E tests as blocking; record unrelated pre-existing failures separately.
- Promote dense CUDA graph batch sizes to defaults only after Engine-side improvements are validated across all online scales.
