# SGLang C++ 重写设计文档

## 1. 目标
使用 C++ (C++23 标准) 重写 [mini-sglang](https://github.com/sgl-project/mini-sglang) Python 项目，利用 LibTorch 进行张量操作，以实现更好的性能和更低的延迟。

## 2. 架构概览
C++ 实现将紧密镜像 Python 架构，但会根据 C++ 习语（静态类型、RAII、智能指针）进行调整。我们将遵循 Google C++ 风格指南并使用其技术栈 (gtest, gflags, glog)。

### 核心组件
1.  **Engine** (`sglang::Engine`): 中央控制器，管理模型、KV 缓存和设备资源。
2.  **Scheduler** (`sglang::Scheduler`): 处理请求调度、批处理以及 Prefill 和 Decode 阶段之间的协调。
3.  **KVCache** (`sglang::kvcache`): 管理用于键值缓存块的 GPU 内存。
4.  **Model** (`sglang::models`): 基于 LibTorch 的模型实现（例如 Llama, Qwen）。
5.  **Context** (`sglang::global_context`): 线程局部或全局上下文。
6.  **Server** (`sglang::server`): 使用 brpc 的 HTTP/RPC 服务器。
7.  **Layers** (`sglang::layers`): 基础构建块 (RoPE, Attention, Linear)。
8.  **Kernels** (`sglang::kernels`): 自定义 CUDA 内核 (例如 Radix 匹配)。
9.  **Tokenizer** (`sglang::tokenizer`): 文本编码/解码。
10. **Distributed** (`sglang::distributed`): 多 GPU 支持 (占位符/未来规划)。
11. **Messages** (`sglang::messages`): 用于 API 通信的数据结构，使用 **Protobuf** 定义 (`protos/sglang.proto`)。
12. **MoE** (`sglang::moe`): 混合专家 (Mixture-of-Experts) 实现。
13. **LLM** (`sglang::llm`): 用于离线推理的高级 API。

## 3. 目录结构
我们将使用 `sglang.cpp` 根目录下的标准 C++ 项目结构。

```text
sglang.cpp/
    ├── CMakeLists.txt
    ├── protos/
    │   └── sglang.proto
    ├── include/
    │   └── sglang/
    │       ├── attention/
    │       ├── core/
    │       ├── distributed/
    │       ├── engine/
    │       ├── kernels/
    │       ├── kvcache/
    │       ├── layers/
    │       │   ├── rotary_embedding.h
    │       │   └── activation.h
    │       ├── llm/
    │       ├── messages/
    │       ├── models/
    │       ├── moe/
    │       ├── scheduler/
    │       ├── server/
    │       ├── tokenizer/
    │       └── utils/
    ├── src/
    │   ├── attention/
    │   ├── core/
    │   ├── distributed/
    │   ├── engine/
    │   ├── kernels/
    │   ├── kvcache/
    │   ├── layers/
    │   ├── llm/
    │   ├── messages/
    │   ├── models/
    │   ├── moe/
    │   ├── scheduler/
    │   ├── server/
    │   ├── tokenizer/
    │   └── main.cpp
    └── tests/
```

## 4. 依赖项
*   **LibTorch (PyTorch C++ API)**: 用于张量操作和神经网络模块。
*   **brpc**: 提供高性能 RPC 和 HTTP 服务能力。
*   **gflags**: 用于命令行参数解析。
*   **glog**: 用于日志记录。
*   **GTest**: 用于单元测试。
*   **nlohmann/json** (可选): 用于配置解析 (倾向于使用 Protobuf 以获得严格模式)。
*   **Protobuf**: 用于定义数据结构和 RPC 消息。
