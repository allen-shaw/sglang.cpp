# scale 0.4 性能瓶颈定位与优化思路

日期：2026-05-05

## 结论摘要

当前 scale `0.4` 的瓶颈不在单个 CUDA 算子本身，也不主要在 GPU 计算吞吐。

更准确的判断是：

- `sglang.cpp` 的主要差距在 host/API 同步和 decode 调度形态。
- `Engine::forward_batch()` 内部的模型算子整体并不比 mini-sglang 慢。
- 低负载下，`sglang.cpp` 可以做到略高吞吐，但 avg/p90 E2E latency 更容易输给 mini。
- 已验证过的粗粒度 scheduler priority 调整可以让 scale `0.4` 单点变好，但会把 latency 转移到 scale `0.8`/`1.0`，不能保留。

因此下一步优化不应该继续做大范围 scheduler 优先级调整，而应该缩小到：

1. output token readback 的同步路径；
2. FlashInfer / graph replay metadata 的 H2D/API 调用路径；
3. 少量残余 GPU kernel gap，例如 rotary，但它不是当前最高优先级。

## Profile 数据来源

主要使用以下已执行 profile / benchmark 结果：

- `docs/profile/nsys/scale04_t6_3/`
  - T6.3 runtime config：CUDA graph batch sizes `1..80,96,112,128`
  - online `/generate`
  - scale `0.4`
  - `256` requests
  - sglang.cpp 与 mini-sglang 使用同一批 synthetic traces
- `docs/profile/release_retest/t6_7_scale04_internal_profile/`
  - `SGLANG_CPP_PROFILE=1`
  - scheduler / engine 内部分段计时
- `docs/profile/release_retest/t6_8_*`
  - T6.8 多个优化 probe 的验证和回滚记录

Release 确认：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
grep 'CMAKE_BUILD_TYPE:STRING=Release' build/CMakeCache.txt
```

## 当前 scale 0.4 表现

nsys 捕获下的 online 对比：

| 指标 | mini-sglang | sglang.cpp | ratio: sglang / mini |
|---|---:|---:|---:|
| req/s | `32.0495` | `32.1749` | `1.0039` |
| tok/s | `4403.55` | `4420.78` | `1.0039` |
| avg E2E | `3.6389 s` | `3.7071 s` | `1.0187` |
| p90 E2E | `5.6384 s` | `5.6901 s` | `1.0092` |

解读：

- 吞吐已经接近或略高于 mini。
- 但请求级 latency 仍偏高。
- 这说明问题不是“算力不够”，而是每个请求在调度、同步和结果回读上的等待形态不够好。

## CUDA API 瓶颈

nsys `cuda_api_sum`：

| CUDA API | sglang.cpp | mini-sglang | 判断 |
|---|---:|---:|---|
| `cudaEventSynchronize` | `6180 ms / 5737 calls` | `5383 ms / 1700 calls` | sglang.cpp event wait 次数明显更多，总等待更高 |
| `cudaMemcpyAsync` | `543 ms / 18840 calls` | `77 ms / 14401 calls` | sglang.cpp H2D/metadata API 开销明显更高 |
| `cudaStreamSynchronize` | `179 ms / 1121 calls` | 非 top item | sglang.cpp 存在额外显式同步压力 |
| `cudaGraphLaunch` | `55 ms / 655 calls` | `52 ms / 703 calls` | graph launch 本身不是主瓶颈 |

关键点：

- `cudaMemcpyAsync` 的 GPU device time 并不高，但 API time 很高。
- 这说明瓶颈更像是 host 侧的小拷贝、tensor copy 调用、事件依赖和 metadata 更新开销，而不是 PCIe 带宽。
- `cudaEventSynchronize` 的调用次数比 mini 多很多，说明 sglang.cpp 仍有较多 host 等 GPU 的同步点。

## GPU kernel 是否是瓶颈

nsys `cuda_gpu_kern_sum` 显示，scale `0.4` 下 sglang.cpp 的主要 kernel 与 mini 基本对齐。

代表性数据：

| 区域 | sglang.cpp | mini-sglang | 判断 |
|---|---:|---:|---|
| 主 GEMM kernel | `752.6 ms / 7728` | `678.1 ms / 7252` | 有差异，但不是单点决定性瓶颈 |
| FFN / CUTLASS kernel | `372.2 ms / 2660` | `414.9 ms / 2996` | sglang.cpp 不慢 |
| FlashInfer attention | `298.2 ms / 4032` | `295.4 ms / 4116` | 基本一致 |
| SiLU activation | `39.9 ms / 4032` | `39.8 ms / 4116` | 基本一致 |
| rotary | `47.7 ms / 4032` | `27.0 ms / 4116` | sglang.cpp 有局部差距，但总量不足以解释 E2E 差距 |
| Q/K RMSNorm | `35.8 ms / 4032` | `26.2 ms / 8232` | fused path 已减少 launch 数，但仍有小差距 |

结论：

- 当前不能简单判断为“Engine forward 的算子慢”。
- 更合理的结论是：GPU kernel 层面大体已对齐，剩余问题主要在 host 同步、metadata 更新和请求调度导致的 E2E 等待。

## 内部分段 profile

`SGLANG_CPP_PROFILE=1` 在 scale `0.4` 下的最终累计输出：

| 分段 | 平均耗时 |
|---|---:|
| `avg_total_us` | `10066.6 us` |
| `avg_schedule_us` | `252.2 us` |
| `avg_prepare_us` | `774.9 us` |
| `avg_forward_us` | `890.3 us` |
| `avg_engine_us` | `874.0 us` |
| `avg_process_us` | `4916.9 us` |
| `avg_process_sync_us` | `4830.6 us` |
| `avg_process_host_us` | `85.7 us` |
| `avg_batch_size` | `45.7` |

关键判断：

- `avg_engine_us` 约 `0.87 ms`，不是最大项。
- `avg_process_sync_us` 约 `4.83 ms`，是最明显的等待项。
- 这对应 `process_forward_output()` 等待 output token copy event 的路径。
- 因此当前瓶颈更偏向 Engine forward 之后的结果同步和调度流水，而不是 forward 算子本身。

## 已验证但不保留的优化方向

T6.8 已做多组可回滚 probe：

| probe | scale 0.4 结果 | 是否保留 | 原因 |
|---|---|---|---|
| 关闭 overlap scheduling | req/tok `0.9876`，avg `1.0377` | 否 | 吞吐和 latency 都变差 |
| workspace event 先 `query()` 再 `synchronize()` | req/tok `1.0010`，avg `1.0221` | 否 | latency 未达标 |
| ready-first overlap processing | req/tok `1.0079`，avg `1.0064` | 否 | latency 未达标 |
| decode burst `2` | scale 0.4 单点通过 | 否 | scale `1.0` avg/p90 回退 |
| low-pressure decode burst, limit `64`/`48` | scale 0.4 有改善 | 否 | scale `0.8`/`1.0` latency 回退 |
| low-pressure decode burst, limit `32`/`40` | scale 0.4 不稳定或失败 | 否 | 不能满足低负载 gate |

结论：

- scheduler 优先级方向确实能影响 scale `0.4`，但太粗。
- 它本质是在不同 scale 之间移动 latency，不是消除根因。
- 因此不应保留这些调度改动。

## 下一步优化思路

### 1. 精确定位 output token readback

当前 `process_sync_us` 是最明显的等待项。

建议目标：

- 细分 `ForwardOutput::synchronize()` 的等待来源；
- 区分等待的是 sampling、D2H copy、token pool writeback，还是下一步 graph/metadata 依赖；
- 避免再次做 T5.7 那种跨多步延迟 readback，因为之前已经触发过 graph/metadata lifetime 问题。

可尝试方向：

- 安全复用 pinned CPU next-token buffer，但生命周期只覆盖当前 one-step overlap；
- 为每个 overlap slot 固定 next-token CPU buffer 和 event；
- 不延迟请求状态更新到多步之后，避免 CUDA graph capture buffer 被提前复用。

验收指标：

- `avg_process_sync_us` 降低；
- `cudaEventSynchronize` 总时间或调用数降低；
- online 三档 scale 不回退。

### 2. 缩小 metadata H2D/API 开销

nsys 显示 `cudaMemcpyAsync` API time 明显高于 mini：

- sglang.cpp：`543 ms / 18840 calls`
- mini-sglang：`77 ms / 14401 calls`

但 GPU memcpy device time 很低：

- sglang.cpp H2D device time：`7.3 ms`
- mini-sglang H2D device time：`13.3 ms`

这说明优化重点不是减少字节量，而是减少 host API 调用成本和小 tensor copy 路径。

建议优先检查：

- `Scheduler::prepare_batch()` 中 positions/mapping/write mapping/write positions 的小 H2D；
- `FlashInferBackend::prepare_for_replay()` 中 graph replay metadata copy；
- `BatchSamplingArgs` 的 per-step sampling metadata；
- 是否存在 pageable CPU tensor 或 ATen copy path 导致 API call 时间异常。

可尝试方向：

- 合并多个 host metadata 数组到一个 pinned struct buffer，一次 H2D；
- 用稳定 device workspace + 小 CUDA kernel 在 device 端展开 metadata；
- 对 decode graph replay 只更新真正变化的字段；
- 对 uniform greedy sampling 再次做更窄的 fast path，但要确认 `/generate` 实际走到该路径。

验收指标：

- `cudaMemcpyAsync` API total 明显下降；
- `cudaMemcpyAsync` call count 不增加；
- `prepare_us` 和 `process_sync_us` 不回退；
- 三档 online gate 同时通过。

### 3. rotary kernel 作为第二优先级

rotary 仍有局部差距：

- sglang.cpp：`47.7 ms / 4032`
- mini-sglang：`27.0 ms / 4116`

但这个差距约 `20 ms`，不足以单独解释 E2E latency 差距。

建议：

- 只有当 readback/metadata 优化没有继续空间时，再优化 rotary。
- 优化方式可以是对齐 mini 的 FlashInfer rotary kernel 参数、layout、head parallelism，或减少额外 layout/view 开销。

### 4. 不建议继续做的方向

短期不建议继续投入：

- 大范围 scheduler decode-priority；
- 关闭 overlap；
- 单纯扩大/缩小 graph batch size；
- 单纯增加 workspace ring size；
- 无 profile 证据的 kernel 重写。

这些方向已经有过 probe，容易造成某个 scale 变好、另一个 scale 回退。

## 当前建议的下一项任务

建议新增下一项任务：

> T6.9：精确拆分 output token readback 和 metadata H2D API 开销。

执行步骤：

1. 在 `ForwardOutput` / `Engine::forward_batch()` / `Scheduler::process_forward_output()` 周围增加更细粒度 profile。
2. 单独统计：
   - sampling kernel 到 D2H copy enqueue 的时间；
   - D2H copy event 等待时间；
   - token pool writeback event 等待时间；
   - FlashInfer metadata H2D copy enqueue 时间。
3. 用 scale `0.4` 跑 profile，确认最大子项。
4. 只针对最大子项做一个可回滚 patch。
5. 每次 patch 必须跑：
   - key E2E correctness；
   - scale `0.4` 单点；
   - scale `0.4,0.8,1.0` 全量。

最终目标不变：

- `req_ratio_sglang_over_mini > 1.0`
- `tok_ratio_sglang_over_mini > 1.0`
- `avg_e2e_ratio_sglang_over_mini <= 1.0`
- `p90_e2e_ratio_sglang_over_mini <= 1.0`

## 2026-05-05 执行优化结果

本轮按上述思路实际尝试了两个局部优化，均已按规则回滚。

### 尝试 1：拆分 readback event 和 completion event

思路：

- 原始实现中，`Engine::forward_batch()` 在 D2H token copy 后记录 `copy_done_event`。
- `Scheduler::forward()` 随后又把同一个 event 重新 record 到 token pool writeback 后。
- 这意味着 `process_forward_output()` 读 CPU token 时等待的不是 D2H readback 完成，而是 D2H + token pool writeback 都完成。
- 因此尝试拆成两个 event：
  - `readback_done_event`：只保护 CPU token readback；
  - `completion_done_event`：保护 token pool writeback 和后续资源复用。

正确性：

- Release build 通过。
- `TestEngineE2E`、`TestSchedulerE2E`、`TestHttpE2E` 通过。

scale `0.4` 结果：

| 指标 | ratio |
|---|---:|
| req/tok | `1.0069` |
| avg E2E | `1.0176` |
| p90 E2E | `1.0058` |

结论：

- 吞吐仍然过线，但 avg/p90 latency 没有过线。
- 说明单独把 CPU readback event 提前，并不能消除主要 E2E 等待；最后的资源同步、请求完成路径或其他 host/API 等待仍会抵消收益。
- 该代码改动已回滚。

### 尝试 2：合并 prepare 阶段 metadata H2D copy

思路：

- 原始 `Scheduler::prepare_batch()` 每步会分别 copy：
  - `positions`
  - `mapping`
  - `write_mapping`
  - `write_positions`
- 尝试把它们合并为两个连续 metadata buffer：
  - token metadata：`positions + mapping`
  - write metadata：`write_mapping + write_positions`
- 目标是把 4 个小 H2D copy 降为 2 个，降低 `cudaMemcpyAsync` API 调用压力。

正确性：

- Release build 通过。
- `TestEngineE2E`、`TestSchedulerE2E`、`TestHttpE2E` 通过。

scale `0.4` 结果：

| 指标 | ratio |
|---|---:|
| req/tok | `1.0097` |
| avg E2E | `1.0150` |
| p90 E2E | `1.0023` |

结论：

- 吞吐有改善，说明减少小 H2D copy 的方向有效。
- 但 avg/p90 latency 仍未达标，尤其 avg E2E 仍高约 `1.5%`。
- 该优化不足以作为保留改动，代码已回滚。

## 更新后的判断

本轮结果进一步收窄了问题范围：

- 单独优化 readback event 不够。
- 单独减少 prepare metadata H2D copy 也不够。
- 当前 latency 差距更可能来自多个小等待叠加：
  - output readback；
  - token pool writeback / resource safety wait；
  - FlashInfer replay metadata；
  - 请求完成时的 detokenize / frontend 聚合路径；
  - CUDA API 小调用的长尾。

下一轮不建议继续做单点微调。更有效的做法是先加更细的 profile，把 `process_sync_us` 拆成：

- readback event wait；
- completion/writeback event wait；
- deferred resource flush wait；
- final request detokenize / frontend wait。

然后只针对最大子项做优化。当前最值得优先验证的是非 streaming `/generate` 场景下的中间 token 处理路径：如果请求设置了 `ignore_eos=true` 且 `stream=false`，理论上中间 token 不需要逐步 detokenize；但之前 T5.7 的跨步 defer readback 触发过 graph/metadata lifetime 问题，所以必须先做生命周期安全设计，再实现。

## 2026-05-05 后续验证：T7.9 - T7.15

本轮继续围绕 scale `0.4` 的 avg E2E latency 做验证。所有性能测试均确认在 Release 构建下执行：

```bash
grep 'CMAKE_BUILD_TYPE:STRING=Release' build/CMakeCache.txt
```

### 当前干净基线复测

`docs/profile/release_retest/t7_12_clean_scale04_profile/online_comparison.csv`

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| `0.4` | `1.0096` | `1.0110` | `0.9980` |

内部 profile 末尾：

| 分段 | 平均耗时 |
|---|---:|
| `avg_prepare_us` | `824.9 us` |
| `avg_forward_us` | `801.4 us` |
| `avg_process_us` | `5076.2 us` |
| `avg_process_sync_us` | `4995.2 us` |
| `avg_process_host_us` | `80.4 us` |

结论：

- scale `0.4` 的 p90 已经接近或略优于 mini。
- 仍未达标的是 avg E2E，主要对应 `process_sync_us`。
- 该等待来自 overlap scheduling 中处理上一轮 output 时等待 GPU 完成；不是 tokenizer/HTTP 的单点 CPU 热点。

### T7.9：Req host page indices / CPU cache values

目标：

- 避免 prefix cache free / evict 路径从 GPU page table 做 D2H 同步。
- 降低 scale `1.0` 下 `allocate_pages_us` 的长尾。

结果：

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| `0.4` | `1.0097` | `1.0118` | `1.0047` |
| `0.8` | `1.0081` | `1.0242` | `0.9977` |
| `1.0` | `1.0089` | `1.0245` | `1.0809` |

结论：能降低部分 cache allocate 同步开销，但端到端 latency 未过线；复测去掉 profile 计时代码后收益不稳定。已回滚，不保留。

### T7.11：非 streaming 不缓存 pending chunk

目标：`/generate stream=false` 不消费中间 chunk，避免每 token 放入 `pending_chunks` 队列。

scale `0.4` 结果：

| req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|
| `1.0055` | `1.0233` | `1.0074` |

结论：明显回退；单纯减少 pending chunk 队列写入不是瓶颈。已回滚。

### T7.12 / T7.14：后移 scheduler 对上一轮 forward done event 的等待

目标：对齐 mini-sglang 的 overlap 形态，让 host scheduling/prepare 更早开始。

尝试内容：

- 把 `last_forward_done_event_` 的等待从 `schedule_next_batch()` 推迟到真正需要读取 `token_pool` 前。
- 或把 input gather 放到 engine stream。

正确性结果：

- `TestEngineE2E` 和 `TestSchedulerE2E` 可通过。
- `TestHttpE2E` 失败，出现 token 串扰和错误输出。

结论：当前 C++ prepare workspace / mapping tensor / token_pool writeback 的生命周期依赖比表面更强。不能简单移动等待点，也不能只把 `token_pool` gather 移到 engine stream。已回滚。

### T7.13：submit coalesce runtime 参数

测试项：

- `SGLANG_CPP_SUBMIT_COALESCE_US=500`
- `SGLANG_CPP_SUBMIT_COALESCE_US=1500`

结果：

- `500us`：scale `0.4` avg E2E ratio `1.0123`，p90 ratio `1.0046`，未达标。
- `1500us`：测试出现卡住/不稳定，不作为候选。

结论：简单调整请求 coalesce 不能解决当前 latency 差距；默认 `1000us` 保持不变。

### T7.15：非 streaming 最终一次性 decode

目标：对 `/generate stream=false` 中间 token 只收集 token id，不逐步 detokenize；请求结束时一次性 decode 全部输出 token。

正确性：`TestTokenizer`、`TestSchedulerE2E`、`TestLLM`、`TestHttpE2E` 通过。

scale `0.4` 结果：

| req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|
| `0.9852` | `1.0456` | `1.0308` |

结论：明显回退。最终集中 decode 拉长请求完成尾部路径，并降低吞吐。已回滚。

## 更新后的瓶颈判断

当前主要瓶颈仍是 scheduler/engine overlap 的同步形态：

- `process_sync_us` 是最大项，约 `5ms/step`。
- 这个等待不是单个 CUDA kernel 慢，而是上一轮 GPU work、token readback、token_pool writeback 与下一轮 scheduler prepare 之间的依赖边界。
- mini-sglang 能把 `token_pool[input_mapping]` 放在 engine stream 上，天然继承上一轮 engine stream 顺序；sglang.cpp 目前在 scheduler stream gather，因此必须更早等待，否则会出现 token 串扰。

下一步更合理的优化方向：

1. 设计完整的 prepare workspace ring：input mapping / positions / write mapping / write positions 均按 slot 复用；每个 slot 有 CUDA event 保护，确保 scheduler 不覆盖 engine 尚未消费的 mapping。
2. 在完整 ring 保护下，把 `token_pool` input gather 放到 engine stream。
3. 先以 `TestHttpE2E` 为正确性 gate，再跑 scale `0.4` 单点，最后跑 `0.4,0.8,1.0` 全量。

短期不建议继续投入：frontend nonstream 路径微调、submit coalesce 参数、单点移动 CUDA event、没有 workspace 生命周期保护的 engine-stream gather。

### T7.16：完整 prepare workspace ring + engine-stream input gather

注意：后续检查发现执行到该阶段时 `build/CMakeCache.txt` 曾被切到 `Debug`，因此本小节中的性能数值不能作为 Release 结论，只保留正确性和方向性记录。已重新执行 Release clean baseline，见 T7.17。

目标：

- 给 prepare 阶段的 mapping / positions / write mapping / write positions 增加 4-slot ring；
- 每个 slot 由 forward completion event 保护，避免 scheduler 覆盖 engine 尚未消费的 workspace；
- 在该生命周期保护下，把 `token_pool` input gather 移到 engine stream；
- 移除 `schedule_next_batch()` 开头对上一轮 `last_forward_done_event_` 的全局等待。

正确性：

- `TestEngineE2E` 通过；
- `TestSchedulerE2E` 通过；
- `TestHttpE2E` 通过。

scale `0.4` 结果：

| req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|
| `1.0058` | `1.0146` | `1.0062` |

结论：

- 完整 ring 可以解决 T7.12/T7.14 的 token 串扰问题；
- 但端到端 latency 仍回退，说明新增 ring slot 管理、额外 event/lifetime 保护和 engine-stream gather 没有带来净收益；
- 该方向正确性可行，但当前实现性能无效，已回滚。

更新建议：

- 暂时不要继续扩大 workspace ring 或只调 ring size；
- 若再次尝试该方向，应先用 nsys 对比 T7.16 与干净基线的 CUDA API/event 数量，确认回退来自 event 同步、额外 gather 排队，还是 graph replay gap。

### T7.17：恢复 Release 后的干净基线复测

执行前重新配置并确认 Release：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
grep 'CMAKE_BUILD_TYPE:STRING' build/CMakeCache.txt
```

确认结果：

```text
CMAKE_BUILD_TYPE:STRING=Release
```

`docs/profile/release_retest/t7_17_clean_release_scale04/online_comparison.csv`

| scale | req/tok ratio | avg E2E ratio | p90 E2E ratio |
|---:|---:|---:|---:|
| `0.4` | `1.0043` | `1.0200` | `1.0077` |

结论：

- Release 干净基线仍未通过 scale `0.4` latency gate；
- 吞吐略高于 mini，但 avg/p90 E2E 均慢；
- 后续所有优化必须先检查 `CMAKE_BUILD_TYPE:STRING=Release`，并将该检查写入 benchmark 记录。
