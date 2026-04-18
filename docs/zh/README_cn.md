# sglang.cpp

<p align="center">
  <img src="../../assets/logo.png" alt="sglang.cpp logo" width="720">
</p>

`sglang.cpp` 是对 [mini-sglang](https://github.com/sgl-project/mini-sglang) 的 C++ 重写版本，目标是在保持整体架构思路对齐的前提下，使用 LibTorch、CUDA 和 C++ 多线程实现更低延迟、更易部署的推理服务。

当前项目已经完成到“控制面 + 接口层”阶段，能够跑通：

- 单卡、单进程推理
- `Engine + Scheduler` 请求调度
- 离线 `LLM` 高级接口
- 基于 `Cinatra` 的 HTTP 服务
- `/generate` 非流式与流式接口
- `/v1/chat/completions` 非流式与流式接口
- 结构化 `messages` 输入的 chat-template 路径

## 当前状态

目前已经具备的能力：

- 已实现模型主链路：`Llama`、`Qwen`、`Qwen3`、`Mistral`、`Qwen3MoE`
- 已实现控制面：`Req` 运行时状态、`Batch`、`Engine`、`Scheduler`
- 已实现资源管理：KV cache、prefix cache、paged table 管理
- 已实现接口层：`ServerArgs`、`FrontendManager`、`SchedulerRunner`、`TokenizerWorkerPool`、`LLM`、`ApiServer`
- 已支持真实 HTTP streaming：基于 `Cinatra` chunked response 输出 SSE
- 已补充真实模型 E2E 测试，包括 “What is the capital of France?” / chat-completions 路径

当前限制：

- 以单机单卡为主
- 主要验证模型为 `Qwen/Qwen3-0.6B`
- 采样参数目前重点支持 `temperature`、`top_k`、`top_p`、`ignore_eos`、`max_tokens/max_new_tokens`
- HTTP 层会接收一部分与 `mini-sglang` 对齐的字段，但并非所有字段都已经在底层生效
- `gflags/glog` 仍在设计依赖列表中，当前代码路径以已有实现为准

## 目录结构

核心目录如下：

```text
sglang.cpp/
├── CMakeLists.txt
├── include/sglang/
│   ├── core/
│   ├── engine/
│   ├── scheduler/
│   ├── models/
│   ├── tokenizer/
│   ├── llm/
│   └── server/
├── src/
│   ├── core/
│   ├── engine/
│   ├── scheduler/
│   ├── models/
│   ├── tokenizer/
│   ├── llm/
│   ├── server/
│   └── main.cpp
├── tests/
└── docs/
```

更多设计说明见：

- [docs/DESIGN_zh.md](../DESIGN_zh.md)
- [docs/DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md)
- [benchmarks/README.md](../../benchmarks/README.md)

## 依赖

项目当前主要依赖：

- LibTorch
- CUDA
- [Cinatra](https://github.com/qicosmos/cinatra)
- [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)
- [nlohmann/json](https://github.com/nlohmann/json)
- GTest

其中一部分依赖通过 `FetchContent` 获取。

## 构建

```bash
cmake -S . -B build -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
cmake --build build -j4
```

构建完成后，主要可执行文件包括：

- `build/sglang_server`
- `build/bench_offline`
- 各类测试程序，例如：
  - `build/test_scheduler_e2e`
  - `build/test_llm`
  - `build/test_http_e2e`

## Benchmark

项目现在已经补充了两类 benchmark：

- 基于 Google Benchmark 的离线 C++ benchmark
- 面向 `/generate` 接口的 Python 在线 benchmark 脚本

具体用法和与 `mini-sglang` 的对比方式见 [../../benchmarks/README.md](../../benchmarks/README.md)。

## 模型准备

当前最常用的验证模型是 `Qwen/Qwen3-0.6B`。可以通过环境变量指定模型目录：

```bash
export QWEN3_MODEL_PATH=/path/to/Qwen3-0.6B
```

如果未设置，测试代码会尝试从 Hugging Face 本地缓存目录中查找：

```text
~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/...
```

## 启动 HTTP 服务

```bash
./build/sglang_server \
  --model-path /path/to/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 1919 \
  --dtype bfloat16 \
  --max-running-requests 4 \
  --max-prefill-length 128 \
  --memory-ratio 0.2
```

如果只想快速验证链路，也可以使用 dummy weight：

```bash
./build/sglang_server \
  --model-path /path/to/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 1919 \
  --dummy-weight
```

## Shell 模式

项目支持本地交互模式：

```bash
./build/sglang_server \
  --model-path /path/to/Qwen3-0.6B \
  --shell-mode
```

进入后可直接输入 prompt，输入 `/exit` 退出。

## HTTP 接口

### 1. `/generate`

非流式：

```bash
curl -X POST http://127.0.0.1:1919/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Question: What is the capital of France?\nAnswer in one short sentence.",
    "max_tokens": 24,
    "ignore_eos": false,
    "stream": false
  }'
```

流式：

```bash
curl -N -X POST http://127.0.0.1:1919/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Question: What is the capital of France?\nAnswer in one short sentence.",
    "max_tokens": 24,
    "ignore_eos": false,
    "stream": true
  }'
```

### 2. `/v1/chat/completions`

非流式：

```bash
curl -X POST http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "What is the capital of France? Answer in one short sentence."}
    ],
    "temperature": 0.0,
    "top_k": -1,
    "top_p": 1.0,
    "max_tokens": 24,
    "stream": false
  }'
```

流式：

```bash
curl -N -X POST http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "What is the capital of France? Answer in one short sentence."}
    ],
    "temperature": 0.0,
    "top_k": -1,
    "top_p": 1.0,
    "max_tokens": 24,
    "stream": true
  }'
```

### 3. `/v1/models`

```bash
curl http://127.0.0.1:1919/v1/models
```

## 离线 LLM 用法

项目中已经实现离线 `LLM` 接口，入口定义在：

- [include/sglang/llm/llm.h](../../include/sglang/llm/llm.h)

它适合做进程内推理、脚本化调用和不经过 HTTP 的集成测试。

## 已验证测试

当前已经补齐并验证过的关键测试包括：

- `test_scheduler_e2e`
  - 单请求
  - 多请求
  - abort
  - chunked prefill
  - prefix cache
  - 真实模型 France/Germany 问答
- `test_llm`
  - dummy weight 下离线 `LLM`
  - 真实模型下 “法国首都” 问答
- `test_http_e2e`
  - `/generate` 非流式
  - `/generate` 流式
  - `/v1/chat/completions` 非流式
  - `/v1/chat/completions` 流式

运行示例：

```bash
./build/test_scheduler_e2e
./build/test_llm
./build/test_http_e2e
```

## 与 `mini-sglang` 的关系

本项目整体思路对齐 `mini-sglang`，但做了几个 C++ 化调整：

- Python 的多进程 + ZMQ，在这里优先采用单进程多线程
- HTTP 服务使用 `Cinatra`
- tokenizer / detokenizer / scheduler 通过进程内对象协作，而不是子进程通信
- `/v1/chat/completions` 路径保留结构化 `messages`，在 tokenizer 层执行 chat-template 渲染

## 后续方向

后续仍计划继续完善：

- 更完整的 OpenAI 兼容字段支持
- 更多模型适配和权重兼容性测试
- 多请求并发 HTTP E2E
- 更强的 streaming 回归测试
- 多卡 / 分布式能力

如果你想继续扩展功能，建议优先查看：

- [docs/DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md)
- [docs/DESIGN_zh.md](../DESIGN_zh.md)
