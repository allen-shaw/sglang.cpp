# sglang.cpp Engine Performance Task List

Date: 2026-04-25

Source plan: `docs/ENGINE_PERFORMANCE_OPTIMIZATION_PLAN.md`

## Gate Policy

Tasks must be completed in order. Do not start the next task until the current task has:

- passed build checks,
- passed key E2E correctness checks,
- produced before/after benchmark or profile data,
- and updated `docs/profile/report.md` or a round-specific note with the result.

All performance benchmarks and profiles after T4.0 must use a Release build. Before every benchmark/profile run:

- run `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`,
- run `cmake --build build -j$(nproc)`,
- verify `build/CMakeCache.txt` contains `CMAKE_BUILD_TYPE:STRING=Release`,
- and record the Release build mode in the round notes.

Debug builds may be used for local correctness debugging only. Debug benchmark results must not be used to accept or reject an optimization.

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
| T5.7 | Rejected | Defer non-streaming host token readback until request completion | Key E2E passed, but online async run hit CUDA illegal memory access due graph/metadata buffer reuse beyond one-step overlap; reverted |
| T5.8 | Rejected | Increase FlashInfer pinned int workspace ring from 32 to 64 | Correctness passed, but online ratios regressed to `0.9817/0.9823/0.9887` and pinned memory doubled; reverted |
| T6.0 | Completed | Re-test post-T4.0 optimization candidates under Release only | Release baseline rerun complete; fused Q/K RMSNorm + narrow candidate retained after Release correctness and benchmark |
| T6.1 | Completed | Release retest T4.2 and T5 rejected attempts | Explicit Release retest completed; partial gains identified for T5.2/T5.6 but no independent attempt passes all-scale gate |
| T6.2 | Rejected | Stack-test T5.6 select-index workspace with retained T6.0 fused Q/K RMSNorm | Release stack improved 0.8 but did not improve 0.4 throughput; reverted |
| T6.3 | Completed | Probe dense graph batch sizes `1..80` with retained T6.0 fused Q/K RMSNorm | Best config so far: scale `0.4` improved to `0.9977`, but still below final gate |
| T6.4 | Rejected | Probe dense graph batch sizes `1..96` with retained T6.0 fused Q/K RMSNorm | Regressed versus T6.3 and T6.0 on scale `0.4`; not retained |
| T6.5 | Rejected | Probe max-running `96` with graph `1..80,96` | Did not improve scale `0.4`; not retained |
| T6.6 | Rejected | Stack-test T5.2 int32 embedding with retained T6.0 and graph `1..80` | Severe scale `0.4` regression; reverted |
| T6.7 | Completed | Focused scale `0.4` nsys comparison with T6.3 config | Bottleneck identified as host/API synchronization and metadata-copy launch overhead, not a single slow model kernel |
| T6.8 | Rejected | Probe scale `0.4` latency-oriented scheduler/readback changes | No probe passed the full all-scale gate; code changes reverted |
| T6.9 | Pending | Close remaining scale `0.4` Release gate across all scales | All three online scales pass final mini-sglang gate in Release |

## Current Task Notes

### T6.0 Release-only retest gate

Problem:

- Round-7 follow-up experiments were initially measured while the active build directory was still `CMAKE_BUILD_TYPE=Debug`.
- Those measurements are useful only as diagnostics and cannot be used to retain or reject performance changes.

Implementation:

- Added a hard Release gate to the task policy and optimization plan.
- Reconfigured and rebuilt the active `build` directory with `-DCMAKE_BUILD_TYPE=Release`.
- Verified `build/CMakeCache.txt` contains `CMAKE_BUILD_TYPE:STRING=Release` before benchmark runs.
- Re-ran the current-best baseline and the fused Q/K RMSNorm + narrow candidate with the same online parameters.

Validation:

- Build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`.
- Verify: `grep 'CMAKE_BUILD_TYPE:STRING=Release' build/CMakeCache.txt`.
- Correctness: `TestNormalization`, `TestEngineE2E`, `TestSchedulerE2E`, and `TestHttpE2E` passed.
- Full Release `ctest` was also run. 25/30 tests passed; the failures were `TestLlama`, `TestMistral`, `TestQwen`, `TestQwen3MoE`, and `TestQwen3Integration`, which remain outside the current online `/generate` acceptance gate.
- Performance artifacts:
  - baseline: `docs/profile/round-7/release_current_best_rerun/`
  - candidate: `docs/profile/round-7/release_fused_qk_rmsnorm_narrow/`

Result:

| scale | baseline req/tok ratio | candidate req/tok ratio | baseline avg E2E ratio | candidate avg E2E ratio | decision |
|---:|---:|---:|---:|---:|---|
| 0.4 | 0.9945 | 0.9953 | 1.0063 | 1.0063 | improved but still below mini |
| 0.8 | 1.0023 | 1.0086 | 0.9898 | 0.9768 | passes final gate |
| 1.0 | 0.9969 | 1.0047 | 1.0082 | 0.9863 | passes final gate |

The fused Q/K RMSNorm + narrow candidate is retained because it improves Release throughput and latency versus the Release baseline at all tested scales. The remaining acceptance blocker is scale `0.4`, where throughput is still about `0.47%` below mini-sglang.

### T6.1 Release retest of T4.2 and T5 rejected attempts

Problem:

- The older T4.2/T5 attempt records did not include per-attempt Release verification output.
- This made the rejection evidence insufficiently auditable.

Implementation:

- Preserved the current T6.0 working state in a temporary stash.
- Returned to the clean T4.0/current-best source state.
- Rebuilt each retest with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`.
- Verified `CMAKE_BUILD_TYPE:STRING=Release` before online benchmark runs.
- Reconstructed missing rejected patches where no git commit/stash existed.
- Reverted each independent probe after measuring it.

Result:

| attempt | scale 0.4 req/tok | scale 0.8 req/tok | scale 1.0 req/tok | decision |
|---|---:|---:|---:|---|
| T4.0 retest baseline | 0.9897 | 1.0013 | 1.0007 | reference |
| T4.2 graph `1..64,72,80,88,96,104,112,120,128` | 0.9862 | 0.9998 | 0.9985 | reject |
| T4.2 graph `1..80,96,112,128` | 0.9917 | 1.0045 | 1.0019 | partial, still fails 0.4 |
| T5.1 graph capture-buffer copy kernel | 0.9913 | 1.0010 | 1.0001 | reject |
| T5.2 int32 embedding | 0.9902 | 1.0028 | 1.0029 | partial, still fails 0.4 |
| T5.3 next-token CPU ring | 0.9916 | 0.9992 | 0.9986 | reject |
| T5.4 uniform-temperature sampler fast path | n/a | n/a | n/a | not exercised by `/generate`; benchmark is greedy |
| T5.5 host page-table mirror | 0.9915 | 1.0020 | 0.9980 | reject |
| T5.6 select-index workspace | 0.9933 | 1.0024 | 1.0018 | partial, best retested T5 candidate |
| T5.8 workspace ring 64 | 0.9907 | 0.9966 | 0.9976 | reject |

Artifacts: `docs/profile/release_retest/`.

Conclusion:

- The user's concern was valid; the old rejection records lacked enough Release proof.
- Under explicit Release retest, T5.2 and T5.6 are not "zero effect"; they produce partial throughput/latency improvements.
- None independently passes the final all-scale mini-sglang gate.
- T5.6 should be stack-tested with the retained T6.0 fused Q/K RMSNorm + narrow optimization before being permanently rejected.

### T6.2-T6.6 stack and config probes

Implementation:

- Re-ran the retained T6.0 fused Q/K RMSNorm + narrow baseline under Release.
- Stack-tested T5.6 select-index workspace with T6.0.
- Probed graph batch-size configs `1..80` and `1..96`.
- Probed `max-running-requests=96`.
- Stack-tested T5.2 int32 embedding with T6.0 and graph `1..80`.
- Reverted rejected code probes after benchmarking.

Result:

| attempt | scale 0.4 req/tok | scale 0.8 req/tok | scale 1.0 req/tok | decision |
|---|---:|---:|---:|---|
| T6.0 fused Q/K baseline | 0.9966 | 1.0037 | 1.0035 | retained |
| T6.2 + T5.6 select-index workspace | 0.9964 | 1.0063 | 1.0035 | rejected |
| T6.3 + graph `1..80,96,112,128` | 0.9977 | 1.0099 | 1.0036 | best config probe, still fails 0.4 |
| T6.4 + graph `1..96,112,128` | 0.9945 | 1.0024 | 1.0031 | rejected |
| T6.5 max-running `96` + graph `1..80,96` | 0.9952 | 1.0067 | 1.0025 | rejected |
| T6.6 + T5.2 int32 embedding + graph `1..80` | 0.9897 | 1.0106 | 1.0022 | rejected |

Artifacts: `docs/profile/release_retest/t6_*`.

Conclusion:

- Keep only the T6.0 code change.
- Use graph `1..80,96,112,128` as the best measured runtime config for the next profile pass, but do not call it final acceptance because scale `0.4` is still below `1.0`.
- Next task is a focused `nsys` comparison at scale `0.4` using the T6.3 config.

### T6.7 focused scale 0.4 profile

Implementation:

- Reconfigured and rebuilt Release.
- Ran aligned `nsys` at online scale `0.4`, `256` requests, T6.3 graph config `1..80,96,112,128`.
- Ran `SGLANG_CPP_PROFILE=1` on the same scale `0.4` shape for scheduler/engine breakdown.

Result:

| metric | sglang.cpp | mini-sglang | conclusion |
|---|---:|---:|---|
| `cudaEventSynchronize` | `6180 ms / 5737` | `5383 ms / 1700` | sglang.cpp has substantially more host waits |
| `cudaMemcpyAsync` API | `543 ms / 18840` | `77 ms / 14401` | host-side async-copy API overhead remains high |
| `cudaGraphLaunch` | `55 ms / 655` | `52 ms / 703` | graph launch itself is not the blocker |
| represented GPU kernel time | about `2037 ms` | about `2081 ms` | model kernels are broadly aligned; sglang.cpp is not slower overall |
| internal `avg_process_sync_us` | `4830.6 us` at 768 steps | n/a | scheduler output processing is dominated by synchronization |

Decision:

- T6.7 is complete. The remaining scale `0.4` problem is primarily host/API synchronization and batching policy, not engine operator throughput.
- T6.8 should be a small, reversible probe targeting low-load latency: output readback synchronization, metadata H2D/API overhead, or low-pressure batch flushing.

### T6.8 latency-oriented probes

All probes were run under Release with T6.3 graph config `1..80,96,112,128`.

Rejected probes:

| probe | scale 0.4 req/tok | scale 0.4 avg | scale 0.4 p90 | all-scale decision |
|---|---:|---:|---:|---|
| disable overlap scheduling | `0.9876` | `1.0377` | `1.0286` | rejected; throughput and latency regressed |
| FlashInfer workspace event `query()` before `synchronize()` | `1.0010` | `1.0221` | `1.0117` | rejected; latency still failed |
| ready-first overlap processing | `1.0079` | `1.0064` | `1.0054` | rejected; latency still failed |
| non-chunked decode burst `2` | `1.0164` | `0.9973` | `0.9942` | rejected; scale `1.0` avg/p90 regressed in all-scale run |
| low-pressure decode burst, batch limit `64` | `1.0132` | `1.0044` | `0.9981` | rejected; scale `0.8`/`1.0` latency regressed |
| low-pressure decode burst, batch limit `48` | `1.0167` | `0.9983` | `0.9959` | rejected; scale `0.8`/`1.0` latency regressed |
| low-pressure decode burst, batch limit `32` | `1.0109` | `1.0113` | `1.0066` | rejected; scale `0.4` latency failed |
| low-pressure decode burst, batch limit `40` | `1.0037` | `1.0208` | `1.0131` | rejected; scale `0.4` latency failed |

Decision:

- No T6.8 code change is retained.
- The best rejected single-scale result was decode burst `2`, but it traded scale `1.0` E2E latency for scale `0.4` gains, so it failed the all-scale acceptance policy.
- Next work should avoid broad scheduler priority changes and instead target a more local source of latency variance, such as output token readback lifetime management or specific metadata-copy call sites.

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

- Build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`.
- Verify: `grep 'CMAKE_BUILD_TYPE:STRING=Release' build/CMakeCache.txt`.
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
- A non-streaming deferred-token-readback fast path skipped per-step CPU token copies for `ignore_eos=true` requests. Key E2E passed and `CUDA_LAUNCH_BLOCKING=1` debug runs completed, but normal async online crashed with CUDA illegal memory access. The failure is consistent with CPU enqueue running beyond the current one-step-overlap safety assumption and reusing CUDA graph/metadata buffers before GPU consumption; reverted.
- Expanding the FlashInfer pinned int workspace ring from 32 to 64 doubled pinned workspace memory and regressed online ratios to `0.9817/0.9823/0.9887`; reverted.
- Sparse graph batch-size probes (`1..64,72,80,88,96,104,112,120,128` and `1..80,96,112,128`) improved some latency numbers but did not improve tok/req ratios enough.
