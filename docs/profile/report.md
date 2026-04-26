# sglang.cpp Profile Report

Date: 2026-04-25

## Environment

- GPU: NVIDIA GeForce RTX 4080 SUPER, 32760 MiB, driver 580.76.05
- Model: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`
- Python: `/root/miniconda3/bin/python`
- mini-sglang: `../mini-sglang`
- Main benchmark: `benchmarks/compare/compare_with_minisgl.py --mode online`

## Baseline

Stable baseline used no-overlap scheduling with CUDA graph capture seq len 1024:

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| 0.4 | 0.8980 | 1.1599 | 1.1442 |
| 0.8 | 0.8935 | 1.2428 | 1.2307 |
| 1.0 | 0.9137 | 1.3079 | 1.2797 |

Default overlap scheduling crashed before this work with CUDA illegal memory access, usually reported from `ForwardOutput::synchronize()`.

## Profile Findings

- Built-in `SGLANG_CPP_PROFILE=1` on no-overlap showed `process_sync_us` as the dominant scheduler cost: about 5.0 ms out of 8.8-9.6 ms per step.
- `nsys` no-overlap CUDA API summary showed `cudaEventSynchronize` at about 64% of CUDA API time, confirming CPU waits on GPU completion.
- Top GPU kernels were expected model kernels: FlashInfer decode/prefill and BF16 GEMMs. No single bad kernel dominated enough to explain the full gap.
- `perf` could not be used because the container had `perf_event_paranoid=4`.
- `heaptrack` was attempted; allocation findings were mostly load/tokenizer side, and the profiled CUDA run hit the same illegal-memory issue before the overlap fix.

## Root Cause Fixed

Two overlap safety issues were fixed:

1. Scheduler tensors produced on `scheduler_stream_` are consumed on `engine_.stream()`. The CUDA caching allocator was not told about the engine stream use, so async overlap could reuse/free storage too early. The scheduler now calls `CUDACachingAllocator::recordStream` for forward input tensors and FlashInfer metadata tensors.
2. FlashInfer reuses one pinned host `pinned_int_workspace_` for decode/prefill plan staging. With overlap, CPU can enqueue the next forward and overwrite that host workspace before the previous async H2D metadata copy has consumed it. The backend now records a CUDA event after workspace-copy users and waits before reusing the host workspace.

After these changes, overlap smoke tests that previously crashed now complete with and without CUDA graph.

## Optimization Rounds

| round | config/result | status |
|---|---|---|
| baseline | no-overlap, graph seq 1024 | stable but 0.89-0.91x mini |
| round-1 | bulk detokenize, coalesce=0, graph bs 128 | no gain; bulk detok reverted |
| round-2 | overlap safety fixes, graph seq 1024 | stable; about 0.90-0.92x mini |
| round-3 | graph capture seq 2048, max bs 128 | best retained config; about 0.94x mini |
| round-3 | dense graph batch sizes | scale 1.0 improved to about 0.95x, but scale 0.4/0.8 still below mini |
| round-4 | remove int32->int64 input cast | negligible gain and worsened compatibility; reverted |
| round-5 / T1.1 | ring-buffered FlashInfer pinned metadata workspace | retained; online ratios improved slightly and `cudaMemcpyAsync` API pressure decreased |
| round-5 / T1.2 attempt | remove replay copy of decode `last_page_len` | correctness passed but perf gate failed; reverted |
| round-5 / T2.1 | align `silu_and_mul` launch with FlashInfer wrapper | retained; activation GPU time dropped from 40.8 ms after T1.1 to 15.3 ms |
| round-5 / T2.2 | Qwen3 running residual with fused-add-RMSNorm | retained; online scale 1.0 ratio improved to 0.9699 |
| round-5 / T3.1 | custom KV cache store kernel | retained; removed ATen `index_copy` and improved scale 1.0 ratio to 0.9783 |
| round-5 / T3.2 attempt | remove RMSNorm forward copy | correctness passed but nsys regressed; reverted |
| round-5 / T3.2 retained | V then Q/K/V strided attention path plus int32 metadata | retained; removed Q/K/V contiguous copies and duplicate int64 metadata |
| round-5 / T3.3 probe | graph capture for every batch size 1..128 | regressed; discarded |
| round-5 / T3.4 | vectorized warp-per-head QK RMSNorm | retained; scale 1.0 ratio improved to 0.9988 before Release |
| round-5 / T3.5 | suppress cinatra debug/trace logging | retained; removes per-connection log I/O |
| round-5 / T3.6 | FlashInfer pinned workspace ring 32 | retained; reduces early pinned workspace reuse waits |
| round-5 / T4.0 | Release build | retained; scale 0.4 and 1.0 throughput exceed mini |
| round-5 / T4.1 probe | decode-priority scheduling | failed SchedulerE2E; reverted |
| round-6 / rotary cache probe | duplicate half-dim rotary cache for FlashInfer v0.2.0 | failed HttpE2E chat-answer checks; reverted |
| round-6 / graph copy kernel | fuse graph replay capture-buffer copies | correctness passed but online ratios regressed; reverted |
| round-6 / int32 embedding | remove input-id int64 cast with custom embedding kernels | correctness passed but throughput gate failed; reverted |
| round-6 / next-token CPU ring | reuse pinned CPU output buffers | correctness passed but scale 0.8/1.0 regressed; reverted |
| round-6 / uniform temperature | skip per-step temperature tensor H2D copy for homogeneous batches | correctness passed but throughput gate failed; reverted |
| round-6 / graph batch-size probes | add sparse graph batch sizes around 64-80 | latency improved in some probes but tok/req gate failed; not retained |
| round-6 / host page table mirror | avoid GPU-to-pageable-CPU page-index copies in cache/free path | correctness passed but online ratios regressed; reverted |
| round-6 / select-index workspace | reuse pinned/device int64 indices for prefill sampling-logit selection | correctness passed but online ratios regressed; reverted |
| round-6 / deferred non-streaming token readback | skip per-step CPU token readback for non-streaming `ignore_eos=true` requests | key E2E passed but async online crashed from unsafe multi-step graph/metadata reuse; reverted |
| round-6 / workspace ring 64 | increase FlashInfer pinned int workspace ring from 32 to 64 | correctness passed but online ratios regressed and pinned memory doubled; reverted |

Best pre-optimization full online result retained in artifacts:

`docs/profile/round-3/online_overlap_graph2048_densebs/online_comparison.csv`

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| 0.4 | 0.9389 | 1.0777 | 1.0686 |
| 0.8 | 0.9427 | 1.1538 | 1.1360 |
| 1.0 | 0.9525 | 1.1567 | 1.1594 |

## Verification

- Build: `cmake --build build -j208` passed.
- Key E2E tests passed during full `ctest`: `TestEngineE2E`, `TestSchedulerE2E`, `TestHttpE2E`, `TestLLM`.
- Full `ctest --test-dir build --output-on-failure` still has 6 failures unrelated to the retained overlap fix: `TestAttentionLayer` segfaults on incomplete-context test, and direct CPU model tests fail because low-level CUDA layers expect CUDA tensors/global context. These failures remain as existing test-suite issues to address separately.

## Current-Best Nsys Alignment

Aligned `nsys` profiles were collected for the current best `sglang.cpp` config and mini-sglang with the same online request shape:

- Artifacts: `docs/profile/nsys/current_best/`
- `sglang.cpp`: `--graph 128 --cuda-graph-capture-max-seq-len 2048 --cuda-graph-bs 1,2,...,64,80,96,112,128`
- workload: online `/generate`, scale `1.0`, 128 synthetic requests, non-streaming

End-to-end result under `nsys`:

| scale | mini tok/s | sglang.cpp tok/s | tok ratio | mini req/s | sglang.cpp req/s | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 3787.02 | 3592.66 | 0.9487 | 26.59 | 25.23 | 1.1527 | 1.1415 |

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

Current bottleneck ranking:

1. Engine host/API overhead around graph replay, metadata staging, and synchronization. The strongest signal is `cudaMemcpyAsync` API time: 312 ms in `sglang.cpp` versus 61 ms in mini-sglang. GPU memory-copy kernel time is tiny, so this is mostly CPU-side API/staging overhead rather than bandwidth.
2. Engine model-path elementwise CUDA kernels. `silu_and_mul`, rotary, and RMSNorm account for about 61 ms extra GPU time versus mini-sglang. Attention and GEMM are close enough that they are not first-order targets.
3. Residual non-graph launch/indexing overhead. `sglang.cpp` still issues more non-graph launches and has visible ATen `index_copy` kernels, but these rank behind metadata/copy API overhead and the slow elementwise kernels.

## Round 5 Task Results

T1.1 replaced the single FlashInfer pinned metadata workspace with a 4-slot ring. This keeps the overlap safety invariant while waiting only on the slot being reused.

Online current-best comparison after T1.1:

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| 0.4 | 0.9504 | 1.0896 | 1.0766 |
| 0.8 | 0.9460 | 1.1423 | 1.1243 |
| 1.0 | 0.9591 | 1.1745 | 1.1666 |

Aligned `nsys` for T1.1 showed `cudaMemcpyAsync` improving from `312 ms / 18926 calls` to `280 ms / 18506 calls`. Total CUDA API time improved slightly from `4388 ms` to `4360 ms`. `cudaStreamSynchronize` time remained noisy and did not resolve, so it stays in the host/API bottleneck bucket.

T1.2 attempted to remove the replay copy of decode `paged_kv_last_page_len`. Correctness passed, but the performance gate failed: aligned `nsys` showed lower `cudaMemcpyAsync` call count but higher `cudaMemcpyAsync` API total time (`319 ms`), and online scale `1.0` was lower than T1.1. The T1.2 code change was reverted.

T2.1 aligned `silu_and_mul` launch configuration with the FlashInfer wrapper by launching one thread per 16-byte vector lane and using `__expf` for SiLU. This was retained.

Online comparison after T2.1:

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| 0.4 | 0.9512 | 1.0558 | 1.0476 |
| 0.8 | 0.9532 | 1.1241 | 1.1075 |
| 1.0 | 0.9566 | 1.1356 | 1.1397 |

Aligned `nsys` showed activation GPU time improving from T1.1 `40.8 ms / 2576 instances` to `15.3 ms / 2688 instances`. Total GPU kernel time improved from T1.1 `1216 ms` to `1207 ms`, and total CUDA API time improved from T1.1 `4360 ms` to `4324 ms`.

T2.2 changed Qwen3 to use mini-sglang's running residual structure with FlashInfer fused-add-RMSNorm. This was retained.

Online comparison after T2.2:

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| 0.4 | 0.9665 | 1.0650 | 1.0501 |
| 0.8 | 0.9651 | 1.1020 | 1.0730 |
| 1.0 | 0.9699 | 1.1075 | 1.1175 |

Aligned `nsys` showed total GPU kernel time improving from T2.1 `1207 ms` to `1180 ms`.

T3.1 replaced the ATen `index_copy_` KV cache write path with a custom CUDA store kernel that writes K and V in one launch and accepts int32/int64 indices. This was retained.

Online comparison after T3.1:

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| 0.4 | 0.9782 | 1.0523 | 1.0474 |
| 0.8 | 0.9706 | 1.0928 | 1.0713 |
| 1.0 | 0.9783 | 1.0862 | 1.1124 |

Aligned `nsys` showed ATen `index_copy` removed and replaced with `store_kv_cache_kernel` at `8.5 ms`. Total GPU kernel time improved to `1174.6 ms`.

T3.2 attempted to remove `RMSNorm::forward`'s copy-before-inplace pattern. Correctness passed, but aligned `nsys` regressed: GPU total increased from T3.1 `1174.6 ms` to `1193.5 ms`, and `direct_copy_kernel_cuda` did not decrease. The T3.2 code change was reverted.

T3.2 retained several copy/indexing reductions:

- V-strided KV store removed the V contiguous copy.
- Q/K/V strided attention removed Q and K contiguous copies and passed Q strides to FlashInfer.
- Scheduler gather/writeback metadata was changed to int32, removing duplicate int64 positions/mapping copies in the hot prepare path.

Online comparison after int32 metadata:

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| 0.4 | 0.9980 | 1.0325 | 1.0213 |
| 0.8 | 0.9886 | 1.0558 | 1.0373 |
| 1.0 | 0.9940 | 1.0523 | 1.0906 |

T3.3 captured all graph batch sizes `1..128` as a probe. It regressed online ratios versus the retained dense policy, so it was discarded.

T3.4 optimized Q/K RMSNorm for strided Q/K tensors with a vectorized warp-per-head kernel. Correctness passed. Aligned `nsys` showed the QK RMSNorm kernel at `21.9 ms`, down from the earlier scalar strided path and closer to mini's `13.7 ms`.

Online comparison after T3.4:

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| 0.4 | 0.9987 | 1.0217 | 1.0198 |
| 0.8 | 0.9921 | 1.0524 | 1.0399 |
| 1.0 | 0.9988 | 1.0414 | 1.0821 |

T3.5 suppressed cinatra debug/trace logging. This removed per-connection debug log lines from online server logs and was retained.

T3.6 expanded the FlashInfer pinned int workspace ring from 4 to 32 slots. Correctness passed. The tradeoff is about 256 MiB pinned host workspace. Online comparison before Release:

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| 0.4 | 0.9998 | 1.0224 | 1.0183 |
| 0.8 | 0.9909 | 1.0556 | 1.0379 |
| 1.0 | 0.9998 | 1.0431 | 1.0756 |

T4.0 found that the active build directory was configured as `CMAKE_BUILD_TYPE=Debug`. Rebuilding as Release is now required for performance measurements.

Release is now a hard gate for all subsequent performance decisions. Each optimization round must reconfigure with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
grep 'CMAKE_BUILD_TYPE:STRING=Release' build/CMakeCache.txt
```

Any benchmark/profile collected from `CMAKE_BUILD_TYPE=Debug` is diagnostic only and must not be used to retain or reject an optimization.

Release current-best online comparison:

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| 0.4 | 1.0024 | 1.0259 | 1.0111 |
| 0.8 | 0.9952 | 1.0447 | 1.0347 |
| 1.0 | 1.0041 | 1.0325 | 1.0807 |

Artifacts: `docs/profile/round-5/t4_0_release_current_best/`.

T4.1 tried decode-priority scheduling to reduce E2E queueing. It failed `TestSchedulerE2E.ChunkedPrefillCanArriveWhileAnotherRequestIsDecoding`, so it was reverted.

## Current Status

The work fixed the main correctness blocker for overlap scheduling and improved online throughput from about 0.91x to roughly parity with mini-sglang. In Release, `sglang.cpp` exceeds mini-sglang throughput at scales `0.4` and `1.0`, but the original full acceptance target is not yet met: scale `0.8` throughput remains below mini, and avg/p90 E2E remains worse at all three scales.

The remaining gap is mostly host/API and queueing latency rather than raw GPU kernel throughput. The next evidence-based targets are:

- Reduce `cudaEventSynchronize` / `cudaMemcpyAsync` API time in `GraphRunner` and `FlashInferBackend` without increasing pinned memory further.
- Improve E2E latency without breaking chunked-prefill semantics; the naive decode-priority attempt was invalid.
- Continue closing the remaining QK RMSNorm and rotary gap against mini's fused kernels.
- Investigate why scale `0.8` remains below mini while scales `0.4` and `1.0` exceed it.

Fresh aligned `nsys` at online scale `0.8` (`docs/profile/round-6/nsys_scale08_current/`) confirmed that the remaining gap is still host/API dominated:

| CUDA API | sglang.cpp | mini-sglang | conclusion |
|---|---:|---:|---|
| `cudaEventSynchronize` | `3300.9 ms / 3940` | `2916.9 ms / 1158` | sglang.cpp waits much more often |
| `cudaMemcpyAsync` | `297.1 ms / 13039` | `53.6 ms / 9640` | sglang.cpp has higher copy API time |
| `cudaStreamSynchronize` | `272.0 ms / 621` | not a top item | caused by small pageable H2D patterns and stream waits |
| `cudaLaunchKernel` + `cudaLaunchKernelEx` | `149.3 ms / 39856` | `227.5 ms / 39439` | kernel launch count is not the largest current gap |
| `cudaGraphLaunch` | `49.7 ms / 477` | `34.3 ms / 486` | sglang.cpp graph launch API cost remains higher |

The same run showed GPU kernel time close enough that isolated host-copy rewrites must still pass the E2E gate. Two targeted attempts, a host KV page-table mirror and a pinned prefill select-index workspace, both reduced plausible API sources in theory but regressed online throughput, so they were reverted.

## Round 6 Rejected Attempts

No new code optimization was retained in round 6.

| attempt | correctness | online result | decision |
|---|---|---|---|
| duplicate rotary cache to full `rotary_dim` | failed `TestHttpE2E` chat checks | not benchmarked | reverted |
| fused graph capture-buffer copy kernel | passed key E2E | ratios `0.9819/0.9913/0.9924` | reverted |
| scalar int32 embedding kernel | passed key E2E | ratios `0.9823/0.9950/0.9949` | reverted |
| vectorized int32 embedding kernel | passed key E2E | ratios `0.9875/0.9876/0.9917` | reverted |
| pinned CPU next-token output ring | passed key E2E | ratios `0.9944/0.9906/0.9895` | reverted |
| uniform-temperature sampler fast path | passed key E2E | ratios `0.9848/0.9918/0.9903` | reverted |
| host KV page-table mirror | passed `TestCacheManager`, `TestEngineE2E`, `TestSchedulerE2E`, `TestHttpE2E` | ratios `0.9927/0.9927/0.9938` | reverted |
| prefill select-index pinned workspace | passed key E2E | ratios `0.9862/0.9953/0.9896` | reverted |
| deferred non-streaming token readback | passed key E2E; blocking debug runs completed | normal async online hit CUDA illegal memory access before completion | reverted |
| FlashInfer workspace ring 64 | passed key E2E | ratios `0.9817/0.9823/0.9887` | reverted |
| graph batch sizes `1..64,72,80,88,96,104,112,120,128` | config-only | ratios `0.9940/0.9942/0.9920` | not retained |
| graph batch sizes `1..80,96,112,128` | config-only | ratios `0.9860/0.9942/0.9918` | not retained |

## Round 7 Release Retest

The initial round-7 follow-up experiments were run while `build/CMakeCache.txt` was still configured as `CMAKE_BUILD_TYPE=Debug`. Those measurements are invalid for performance decisions. Round 7 is being restarted with an explicit Release gate:

- Configure: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- Build: `cmake --build build -j$(nproc)`
- Verify: `grep 'CMAKE_BUILD_TYPE:STRING=Release' build/CMakeCache.txt`
- Then rerun current-best baseline and each candidate optimization under the same Release configuration.

Release gate completed on 2026-04-26:

- Build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`.
- Verification: `CMAKE_BUILD_TYPE:STRING=Release`.
- Correctness for the candidate: `TestNormalization`, `TestEngineE2E`, `TestSchedulerE2E`, and `TestHttpE2E` passed.
- Full Release `ctest --test-dir build --output-on-failure` result: 25/30 tests passed. Failures were `TestLlama`, `TestMistral`, `TestQwen`, `TestQwen3MoE`, and `TestQwen3Integration`; these are outside the online `/generate` acceptance path and are not treated as blocking by the current performance gate. The blocking key tests still passed.

Release current-best baseline rerun:

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| 0.4 | 0.9945 | 1.0063 | 1.0072 |
| 0.8 | 1.0023 | 0.9898 | 0.9942 |
| 1.0 | 0.9969 | 1.0082 | 1.0083 |

Artifacts: `docs/profile/round-7/release_current_best_rerun/`.

Release fused Q/K RMSNorm + narrow candidate:

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| 0.4 | 0.9953 | 1.0063 | 1.0047 |
| 0.8 | 1.0086 | 0.9768 | 0.9784 |
| 1.0 | 1.0047 | 0.9863 | 0.9839 |

Artifacts: `docs/profile/round-7/release_fused_qk_rmsnorm_narrow/`.

Decision:

- Retain the fused Q/K RMSNorm + narrow candidate. It improves Release throughput and latency versus the Release baseline at all tested scales.
- The original final acceptance target is still not fully met because scale `0.4` remains below mini-sglang on throughput by about `0.47%`.
- Next target: profile or micro-benchmark the low-concurrency scale `0.4` path specifically; avoid using Debug measurements for this decision.

## Release Retest of T4.2 and T5 Rejected Attempts

The earlier T4.2/T5 records did not contain a per-attempt `CMAKE_BUILD_TYPE` proof line, so the attempts below were rebuilt and re-run under an explicit Release gate on 2026-04-26:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
grep 'CMAKE_BUILD_TYPE:STRING=Release' build/CMakeCache.txt
```

Baseline for this retest batch: `docs/profile/release_retest/t4_0_baseline/`.

| attempt | scale 0.4 req/tok | scale 0.8 req/tok | scale 1.0 req/tok | scale 0.4 avg | scale 0.8 avg | scale 1.0 avg | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| T4.0 baseline | 0.9897 | 1.0013 | 1.0007 | 1.0137 | 0.9892 | 0.9981 | reference |
| T4.2 graph `1..64,72,80,88,96,104,112,120,128` | 0.9862 | 0.9998 | 0.9985 | 1.0176 | 1.0007 | 1.0057 | reject |
| T4.2 graph `1..80,96,112,128` | 0.9917 | 1.0045 | 1.0019 | 1.0103 | 0.9854 | 0.9867 | partial, still fails 0.4 gate |
| T5.1 fused graph capture-buffer copy kernel | 0.9913 | 1.0010 | 1.0001 | 1.0142 | 0.9971 | 1.0013 | reject |
| T5.2 int32 embedding kernel | 0.9902 | 1.0028 | 1.0029 | 1.0159 | 0.9885 | 0.9836 | partial, still fails 0.4 gate |
| T5.3 pinned CPU next-token output ring | 0.9916 | 0.9992 | 0.9986 | 1.0094 | 0.9995 | 1.0072 | reject |
| T5.4 uniform-temperature sampler fast path | n/a | n/a | n/a | n/a | n/a | n/a | not exercised by `/generate`; server forces greedy `temperature=0` for this benchmark |
| T5.5 host KV page-table mirror | 0.9915 | 1.0020 | 0.9980 | 1.0108 | 0.9959 | 1.0057 | reject |
| T5.6 prefill select-index workspace | 0.9933 | 1.0024 | 1.0018 | 1.0133 | 0.9953 | 0.9916 | partial, still fails 0.4 gate |
| T5.8 FlashInfer workspace ring 64 | 0.9907 | 0.9966 | 0.9976 | 1.0115 | 1.0075 | 1.0064 | reject |

Artifacts are under `docs/profile/release_retest/`.

Conclusion:

- The suspicion was valid: without per-attempt Release proof, the older records were not sufficiently auditable.
- Re-running under Release shows that several attempts are not strictly "no effect"; T4.2 `1..80`, T5.2, and T5.6 have partial gains versus this retest baseline.
- None of the independent attempts satisfies the final all-scale gate because scale `0.4` remains below mini-sglang and has worse avg E2E.
- T5.6 is the most useful candidate to stack-test with the retained T6.0 fused Q/K RMSNorm + narrow optimization.

## Round 7 Stack Tests After Release Retest

The current retained code baseline is T6.0: fused Q/K RMSNorm + `narrow` in attention. It was re-run before stack tests:

| attempt | scale 0.4 req/tok | scale 0.8 req/tok | scale 1.0 req/tok | scale 0.4 avg | scale 0.8 avg | scale 1.0 avg | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| T6.0 fused Q/K baseline | 0.9966 | 1.0037 | 1.0035 | 1.0057 | 0.9885 | 0.9868 | retained |
| T6.2 T6.0 + T5.6 select-index workspace | 0.9964 | 1.0063 | 1.0035 | 1.0022 | 0.9821 | 0.9859 | rejected; 0.4 throughput did not improve |
| T6.3 T6.0 + graph `1..80,96,112,128` | 0.9977 | 1.0099 | 1.0036 | 1.0001 | 0.9730 | 0.9887 | best config probe, still fails 0.4 throughput gate |
| T6.4 T6.0 + graph `1..96,112,128` | 0.9945 | 1.0024 | 1.0031 | 1.0105 | 0.9962 | 0.9917 | rejected |
| T6.5 T6.0 + max-running `96` + graph `1..80,96` | 0.9952 | 1.0067 | 1.0025 | 1.0041 | 0.9813 | 0.9907 | rejected |
| T6.6 T6.0 + T5.2 int32 embedding + graph `1..80` | 0.9897 | 1.0106 | 1.0022 | 1.0157 | 0.9729 | 0.9942 | rejected; severe 0.4 regression |

Artifacts: `docs/profile/release_retest/t6_*`.

Conclusion:

- The only retained code change remains T6.0 fused Q/K RMSNorm + `narrow`.
- T6.3 graph `1..80,96,112,128` is the best measured runtime configuration so far, improving scale `0.4` from `0.9966` to `0.9977`, but it still does not satisfy the final `>1.0` throughput gate.
- T5.6 and T5.2 should not be stacked by default; both failed the scale `0.4` decision criterion when combined with T6.0/T6.3.
- The remaining gap is now too small for broad host-copy guesses. Next work should collect a focused `nsys` comparison at scale `0.4` using the T6.3 config and inspect the residual host/API difference.
