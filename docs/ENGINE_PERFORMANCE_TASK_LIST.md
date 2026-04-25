# sglang.cpp Engine Performance Task List

Date: 2026-04-25

Source plan: `docs/ENGINE_PERFORMANCE_OPTIMIZATION_PLAN.md`

## Gate Policy

Tasks must be completed in order. Do not start the next task until the current task has:

- passed build checks,
- passed key E2E correctness checks,
- produced before/after benchmark or profile data,
- and updated `docs/profile/report.md` or a round-specific note with the result.

Primary acceptance remains online `/generate` against `mini-sglang` at scales `0.4`, `0.8`, and `1.0`:

- `tok_ratio_sglang_over_mini > 1.0`
- `req_ratio_sglang_over_mini > 1.0`
- `avg_e2e_ratio_sglang_over_mini <= 1.0`
- `p90_e2e_ratio_sglang_over_mini <= 1.0`

## Task Queue

| ID | Status | Task | Success Gate |
|---|---|---|---|
| T1.1 | Completed | Replace single FlashInfer pinned metadata workspace with a ring-buffered workspace | Build passed; key E2E tests passed; online ratios improved; `cudaMemcpyAsync` improved from `312 ms / 18926 calls` to `280 ms / 18506 calls` |
| T1.2 | Pending | Reduce graph replay metadata `cudaMemcpyAsync` count in `prepare_for_replay` | `cudaMemcpyAsync` call count and API time decrease versus `docs/profile/nsys/current_best` |
| T1.3 | Pending | Reduce `GraphRunner::replay` host overhead and graph launch overhead | `cudaGraphLaunch` average API cost moves closer to mini-sglang |
| T2.1 | Completed | Replace or port faster `silu_and_mul` kernel path | activation kernel time improved from `40.8 ms` after T1.1 to `15.3 ms`; correctness passed |
| T2.2 | Completed | Add faster/fused RMSNorm/qk-norm/fused-add-RMSNorm paths | Online scale `1.0` ratio improved to `0.9699`; aligned `nsys` GPU total dropped to `1179.7 ms` |
| T2.3 | Pending | Optimize rotary kernel path/layout | rotary total time approaches mini-sglang |
| T3.1 | Completed | Replace hot KV-cache `index_copy_` path with custom CUDA store kernel | `index_copy` disappeared; online scale `1.0` ratio improved to `0.9783` |
| T3.2 | Completed | Remove remaining hot ATen indexing/copy ops from decode steady state | V-strided KV store, Q/K/V strided attention, and int32 metadata path retained; direct copy/H2D pressure decreased |
| T3.3 | Rejected | Expand CUDA graph coverage to every batch size 1..128 | Probe regressed online ratios; keep dense selected batch-size policy |
| T3.4 | Completed | Optimize Q/K RMSNorm strided path | Vectorized warp-per-head QK RMSNorm retained; scale `1.0` ratio improved to `0.9988` before Release |
| T3.5 | Completed | Suppress hot-path HTTP debug/trace logging | Per-connection cinatra debug logging removed; correctness passed; retained |
| T3.6 | Completed | Reduce FlashInfer pinned workspace reuse waits | Ring expanded from 4 to 32; correctness passed; Release scale `1.0` ratio reached `1.0041` |
| T4.0 | Completed | Rebuild and benchmark Release configuration | Release build passed; tok/req exceeded mini at scales `0.4` and `1.0`, but full E2E gate still not met |
| T4.1 | Rejected | Decode-priority scheduler probe for lower E2E | Broke chunked-prefill SchedulerE2E; reverted |
| T4.2 | Rejected | Add sparse extra CUDA graph batch sizes around 64-80 | Latency improved in some probes, but tok/req ratios stayed below current-best; no default/config change retained |
| T5.1 | Rejected | Fuse graph replay input/out_loc/positions capture-buffer copies into one CUDA kernel | Correctness passed, but online ratios regressed; reverted |
| T5.2 | Rejected | Replace input-id int32-to-int64 cast with custom int32 embedding kernel | Correctness passed, but scalar/vectorized kernels failed throughput gate; reverted |
| T5.3 | Rejected | Reuse pinned CPU next-token output buffers with a ring | Correctness passed, but scale `0.8`/`1.0` throughput and latency regressed; reverted |
| T5.4 | Rejected | Avoid per-step temperature tensor H2D copy for uniform-temperature batches | Correctness passed, but online tok/req ratios regressed; reverted |
| T5.5 | Rejected | Mirror KV page table on host to avoid device-to-pageable-CPU page-index copies | Correctness passed, but online tok/req ratios regressed to `0.9927/0.9927/0.9938`; reverted |
| T5.6 | Rejected | Reuse pinned/device index workspace for prefill `select_sampling_logits` | Correctness passed, but online tok/req ratios regressed to `0.9862/0.9953/0.9896`; reverted |

## Current Task Notes

### T1.1 Ring-buffer FlashInfer pinned metadata workspace

Problem:

- Current `FlashInferBackend` reuses one pinned host `pinned_int_workspace_`.
- Overlap safety requires waiting before reusing that host buffer.
- Current-best `nsys` shows extra synchronization/API pressure: `cudaStreamSynchronize` is `210 ms / 613 calls`, and `cudaMemcpyAsync` is `312 ms / 18926 calls`.

Implementation:

- Allocate a fixed ring of pinned CPU workspaces with the same size as the existing workspace.
- Track one CUDA event per ring slot.
- Before writing a slot, wait only for that slot's prior async copy user, not for all prior workspace use.
- Record completion on the slot after enqueueing each async H2D workspace copy or plan operation using that slot.
- Keep the existing correctness guarantee: host workspace memory must not be overwritten before the GPU has consumed the previous async copy from that slot.

Validation:

- Build: `cmake --build build -j$(nproc)`.
- Correctness: key E2E tests for engine/scheduler/http.
- Performance: run the same online comparison and aligned `nsys` current-best workflow, store artifacts in `docs/profile/round-5/`.

Result:

- Implemented a 4-slot pinned metadata workspace ring.
- Build passed.
- Key E2E tests passed: `TestEngineE2E`, `TestSchedulerE2E`, `TestHttpE2E`, `TestLLM`.
- Online current-best comparison improved all tested scales:
  - scale `0.4`: tok/req ratio `0.9504`
  - scale `0.8`: tok/req ratio `0.9460`
  - scale `1.0`: tok/req ratio `0.9591`
- Aligned `nsys` showed lower `cudaMemcpyAsync` API pressure for `sglang.cpp`: `312 ms / 18926 calls` to `280 ms / 18506 calls`.
- `cudaStreamSynchronize` total time rose in this `nsys` run, so sync pressure remains a T1.2/T1.3 follow-up instead of being considered resolved.

### T1.2 Reduce replay metadata copies

Problem:

- `prepare_for_replay` still copies several metadata tensors into capture buffers every decode step.
- Current-best `nsys` after T1.1 still shows `cudaMemcpyAsync` as a top API item: `280 ms / 18506 calls`.

Implementation:

- Audit each replay metadata copy and classify it as required per step, static per captured batch size, or derivable on device.
- Remove or fuse per-step copies that can be replaced by an existing scheduler-produced tensor or a small custom update kernel.
- Preserve graph replay correctness: captured pointers must remain stable, and replay-visible device data must be updated before `CUDAGraph::replay`.

Validation:

- Same build, key E2E, online comparison, and aligned `nsys` checks as T1.1.

Attempt log:

- Attempted to remove the per-step replay copy of `paged_kv_last_page_len`, because decode uses `page_size == 1` and the value is always one.
- Correctness passed, but performance gate failed:
  - `cudaMemcpyAsync` call count decreased, but API total increased to `319 ms` in the aligned `nsys` run.
  - Online scale `1.0` was lower than T1.1.
- The change was reverted. T1.2 remains open and should target larger metadata-copy consolidation rather than deleting this single copy in isolation.

### T2.1 Align `silu_and_mul` with FlashInfer wrapper launch shape

Problem:

- Current-best `nsys` shows `silu_and_mul` at `42.8 ms`, while mini-sglang is at `14.4 ms`.
- mini-sglang calls FlashInfer's Python/CUDA wrapper, whose C++ launcher uses one thread per 16-byte vector lane: `block = min(d / vec_size, 1024)`.
- `sglang.cpp` directly instantiates the same FlashInfer kernel but launched with `block = min(d, 1024)`, which creates about 8x more threads for FP16/BF16 hidden dimensions.

Implementation:

- Align `silu_and_mul` launch configuration with FlashInfer's wrapper.
- Use `__expf` for SiLU to match the FlashInfer wrapper's fast math path.
- Apply the same vectorized block sizing to `gelu_and_mul` for consistency, although Qwen3 uses SiLU.

Validation:

- Build and run `TestActivation` plus key E2E tests.
- Run online current-best comparison.
- Run aligned `nsys`; success requires activation GPU time to decrease without E2E regression.

Result:

- Aligned launch configuration with FlashInfer's wrapper by using one thread per 16-byte vector lane.
- Switched SiLU from `expf` to `__expf`, matching FlashInfer's fast path.
- Build passed.
- `TestActivation`, `TestEngineE2E`, `TestSchedulerE2E`, `TestLLM`, and `TestHttpE2E` passed.
- Online comparison after T2.1:
  - scale `0.4`: tok/req ratio `0.9512`
  - scale `0.8`: tok/req ratio `0.9532`
  - scale `1.0`: tok/req ratio `0.9566`
- Aligned `nsys` showed activation GPU time improving from T1.1 `40.8 ms` to `15.3 ms`, close to mini-sglang's previous `14.4 ms`.
- Total GPU kernel time improved from T1.1 `1216 ms` to `1207 ms`.

### T2.2 Faster RMSNorm/fused norm path

Problem:

- Current-best `nsys` shows RMSNorm family at about `47 ms`, while mini-sglang is about `33 ms`.
- mini-sglang uses faster fused norm kernels for fused-add-RMSNorm and qk-RMSNorm paths.

Implementation:

- Inspect current RMSNorm and attention q/k norm call sites.
- First try launch-shape or wrapper alignment if the current code directly instantiates FlashInfer kernels differently from mini-sglang.
- If wrapper alignment is insufficient, add fused kernels only for the model paths that are visible in nsys.

Validation:

- Build and run `TestNormalization`, attention/model E2E tests, online comparison, and aligned `nsys`.
- Retain only if RMSNorm GPU time decreases without online regression.

Result:

- Added `RMSNorm::fused_add_forward_inplace`.
- Changed Qwen3 residual flow to match mini-sglang's running-residual `fused_add_rmsnorm` path.
- Correctness passed: `TestNormalization`, `TestEngineE2E`, `TestSchedulerE2E`, `TestLLM`, and `TestHttpE2E`.
- Online comparison after T2.2:
  - scale `0.4`: tok/req ratio `0.9665`
  - scale `0.8`: tok/req ratio `0.9651`
  - scale `1.0`: tok/req ratio `0.9699`
- Aligned `nsys` showed total GPU kernel time improving to `1179.7 ms`, close to mini-sglang's current run.

### T3.1 Custom KV cache store kernel

Problem:

- KV cache writes used two ATen `index_copy_` kernels per layer and converted `out_loc` to int64.
- Current `nsys` showed `index_copy` around `20.7 ms` after T2.2.

Implementation:

- Replaced `MHAKVCache::store_kv` with a CUDA kernel that writes K and V in one launch.
- Supports int32 and int64 `out_loc` without hot-path dtype conversion.

Validation:

- Build passed.
- `TestCacheManager`, `TestEngineE2E`, `TestSchedulerE2E`, `TestLLM`, and `TestHttpE2E` passed.
- Online comparison after T3.1:
  - scale `0.4`: tok/req ratio `0.9782`
  - scale `0.8`: tok/req ratio `0.9706`
  - scale `1.0`: tok/req ratio `0.9783`
- Aligned `nsys` showed ATen `index_copy` removed and replaced with `store_kv_cache_kernel` at `8.5 ms`.

### T3.2 Attempt: remove RMSNorm forward copy

Attempt:

- Removed `RMSNorm::forward`'s `empty_like + copy_ + forward_inplace` sequence and called out-of-place FlashInfer RMSNorm directly.

Result:

- Correctness passed, but performance gate failed.
- Online comparison was mostly neutral, but aligned `nsys` regressed from T3.1:
  - GPU total increased from `1174.6 ms` to `1193.5 ms`.
  - `direct_copy_kernel_cuda` did not decrease.
  - `cudaMemcpyAsync` and launch count increased.
- The change was reverted. T3.2 remains open.

### T3.2 Retained: strided attention and int32 metadata

Results:

- Removed the V contiguous copy first by teaching the custom KV store kernel to accept strided V.
- Extended the path to Q/K/V strided inputs and passed Q strides into FlashInfer attention params.
- Switched scheduler gather/writeback metadata from int64 to int32, removing the duplicate int64 positions copy from prepare.
- Correctness passed: `TestCacheManager`, `TestNormalization`, `TestEngineE2E`, `TestSchedulerE2E`, `TestLLM`, and `TestHttpE2E`.
- Online ratio after Q/K/V strided path reached scale `1.0` `0.9914`; after int32 metadata it was scale `1.0` `0.9940`.

### T3.3 Rejected: graph all batch sizes

Attempt:

- Captured CUDA graphs for every batch size `1..128`.

Result:

- Online performance regressed versus the retained dense policy `1..64,80,96,112,128`.
- The probe was discarded; no code/config default was changed.

### T3.4 Vectorized Q/K RMSNorm

Problem:

- After Q/K/V strided attention, custom scalar strided QK RMSNorm was visible in `nsys`.
- Mini-sglang used a vectorized QK RMSNorm kernel and was faster.

Implementation:

- Replaced the scalar strided QK RMSNorm hot path with a warp-per-head vectorized kernel using 16-byte `flashinfer::vec_t<T, 8>` load/store when `head_dim` is divisible by 8.
- Kept the scalar kernel as fallback.

Result:

- Correctness passed.
- QK RMSNorm nsys time improved from `49.5 ms` initially to `21.9 ms`.
- Online scale `1.0` ratio improved to `0.9988` before Release.

### T3.5 HTTP logging

Problem:

- cinatra emitted per-connection debug logs in Debug builds.

Implementation:

- Added compile definitions to map `CINATRA_LOG_DEBUG` and `CINATRA_LOG_TRACE` to `cinatra::NULL_LOGGER`.

Result:

- Correctness passed.
- Per-connection log lines disappeared from server logs.
- Retained because it removes hot-path I/O noise with no behavior change.

### T3.6 FlashInfer pinned workspace ring 32

Problem:

- `nsys` still showed high `cudaEventSynchronize` API time.
- The 4-slot pinned metadata ring could be reused within one model forward, forcing host waits before overwriting pinned staging memory.

Implementation:

- Increased pinned int workspace ring size from 4 to 32.

Result:

- Correctness passed.
- Online scale `1.0` ratio improved to `0.9998` before Release.
- Memory tradeoff: pinned host workspace increased from about 32 MiB to 256 MiB.

### T4.0 Release Build

Problem:

- The build directory was configured as `CMAKE_BUILD_TYPE=Debug`, which invalidates host-side performance conclusions.

Implementation:

- Reconfigured with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`.

Result:

- Correctness passed: `TestNormalization`, `TestCacheManager`, `TestEngineE2E`, `TestSchedulerE2E`, `TestLLM`, `TestHttpE2E`.
- Online Release current-best:
  - scale `0.4`: tok/req ratio `1.0024`, avg E2E ratio `1.0259`, p90 E2E ratio `1.0111`
  - scale `0.8`: tok/req ratio `0.9952`, avg E2E ratio `1.0447`, p90 E2E ratio `1.0347`
  - scale `1.0`: tok/req ratio `1.0041`, avg E2E ratio `1.0325`, p90 E2E ratio `1.0807`
- Full acceptance is not met yet because scale `0.8` throughput remains below mini and E2E latency remains worse.

### T4.1 Rejected: decode-priority scheduling

Attempt:

- Tried scheduling decode before prefill to reduce online E2E queueing.

Result:

- `TestSchedulerE2E.ChunkedPrefillCanArriveWhileAnotherRequestIsDecoding` failed.
- The change was reverted.

### Round 6 rejected probes

Attempted and reverted or discarded:

- Fresh aligned `nsys` at scale `0.8` still points to host/API overhead: `cudaEventSynchronize` is `3300.9 ms / 3940` for sglang.cpp versus `2916.9 ms / 1158` for mini, `cudaMemcpyAsync` is `297.1 ms / 13039` versus `53.6 ms / 9640`, and `cudaStreamSynchronize` remains a top item only for sglang.cpp.
- Rotary cache duplication for FlashInfer v0.2.0 changed chat-completion output distribution and failed `TestHttpE2E`.
- A fused CUDA kernel for graph replay capture-buffer updates replaced three small copies with one launch. Correctness passed, but online ratios regressed to `0.9819/0.9913/0.9924`.
- A custom int32 embedding path removed the explicit input-id int64 cast. Correctness passed, but scalar and vectorized variants failed throughput gates; best scalar run was `0.9823/0.9950/0.9949`.
- A 4-slot pinned CPU next-token output buffer ring reduced allocation churn, but scale `0.8`/`1.0` regressed.
- A uniform-temperature sampler fast path avoided one H2D copy for homogeneous sampling params. Correctness passed, but online ratios were `0.9848/0.9918/0.9903`; reverted.
- A host-side KV page-table mirror removed the direct need to copy page indices back from GPU when caching/freeing requests. Correctness passed, but online ratios were `0.9927/0.9927/0.9938`; reverted.
- A pinned/device index workspace for prefill sampling-logit selection targeted small pageable H2D copies before stream sync. Correctness passed, but online ratios were `0.9862/0.9953/0.9896`; reverted.
- Sparse graph batch-size probes (`1..64,72,80,88,96,104,112,120,128` and `1..80,96,112,128`) improved some latency numbers but did not improve tok/req ratios enough.
