
# Mini-SGLang C++ 重写设计文档

## 1. 目标
使用 C++ (C++23 标准) 重写 [mini-sglang](https://github.com/sgl-project/mini-sglang) Python 项目，利用 LibTorch 进行张量操作，以实现更好的性能和更低的延迟。Python 实现的核心逻辑（RadixAttention、Continuous Batching）将被保留。

## 2. 架构概览
C++ 实现将紧密镜像 Python 架构，但会根据 C++ 习语（静态类型、RAII、智能指针）进行调整。我们将遵循 Google C++ 风格指南并使用其技术栈 (gtest, gflags, glog)。

### 核心组件
1.  **Engine** (`sglang::Engine`): 中央控制器，管理模型、KV 缓存和设备资源。
2.  **Scheduler** (`sglang::Scheduler`): 处理请求调度、批处理以及 Prefill 和 Decode 阶段之间的协调。
3.  **KVCache** (`sglang::kvcache`): 管理用于键值缓存块的 GPU 内存。
    *   `RadixCacheManager` & `RadixTreeNode`: 使用基数树（Radix Tree）实现前缀缓存。
    *   `CacheManager`: 用于内存分配的抽象接口。
4.  **Model** (`sglang::models`): 基于 LibTorch 的模型实现（例如 Llama, Qwen）。
5.  **Context** (`sglang::global_context`): 线程局部或全局上下文，用于在执行期间管理请求状态。
6.  **Server** (`sglang::server`): 使用 **brpc** 处理 HTTP/RPC 请求，替代 Python 的 FastAPI。

## 3. 目录结构
我们将使用 `sglang.cpp` 根目录下的标准 C++ 项目结构。

```text
sglang.cpp/
├── CMakeLists.txt
├── include/
│   └── sglang/
│       ├── core/
│       │   ├── batch.h
│       │   ├── context.h
│       │   └── req.h
│       ├── engine/
│       │   ├── config.h
│       │   ├── engine.h
│       │   └── tensor_utils.h
│       ├── kvcache/
│       │   ├── cache_manager.h
│       │   └── radix_tree.h
│       ├── models/
│       │   └── llama.h
│       ├── scheduler/
│       │   ├── scheduler.h
│       │   ├── prefill.h
│       │   └── decode.h
│       ├── server/
│       │   └── http_server.h
├── src/
│   ├── core/
│   ├── engine/
│   ├── kvcache/
│   ├── models/
│   ├── scheduler/
│   └── main.cpp
└── tests/
```

## 4. 详细组件设计

### 4.1 核心数据结构 (`include/sglang/core/`)
*   **Req**: 保存请求状态的结构体 (`input_ids`, `output_ids`, `sampling_params`, `cache_handle`, `uid`)。
    *   使用 `std::shared_ptr<Req>` 进行生命周期管理。
*   **Batch**: 保存 `Req` 向量和阶段信息（`PREFILL`, `DECODE`）的结构体。

### 4.2 KV 缓存 (`include/sglang/kvcache/`)
*   **RadixTreeNode**:
    *   表示前缀树中的一个节点。
    *   保存 `key`（Token）和 `value`（KV 缓存中的索引）。
    *   如果需要，使用 `std::shared_mutex` 进行线程安全访问（最初假设单调度器）。
*   **RadixCacheManager**:
    *   管理基数树。
    *   实现 `match_prefix`, `insert_prefix`, `evict`。

### 4.3 调度器 (`include/sglang/scheduler/`)
*   **Scheduler**:
    *   拥有 `Engine`, `CacheManager`。
    *   `run_loop()`: 通过轮询或条件变量处理请求的主循环。
    *   `schedule_next_batch()`: 从等待队列中选择请求。
*   **PrefillManager**: 处理新请求。
*   **DecodeManager**: 处理正在运行的请求。

### 4.4 引擎 (`include/sglang/engine/`)
*   **Engine**:
    *   初始化 `torch::Device`。
    *   加载模型权重（使用 `torch::load` or safetensors 适配器）。
    *   `forward_batch(Batch& batch)`: 使用 `torch::jit` 或自定义 `nn::Module` 执行模型。

### 4.5 服务器 (`include/sglang/server/`)
*   **HttpServer**:
    *   使用 **brpc** (Baidu RPC, 广泛使用的工业级 C++ RPC 框架，兼容 gflags/glog/protobuf) 处理 HTTP/REST 请求。
    *   利用 brpc 的高性能 M:N 协程模型 (bthread) 替代 FastAPI 的异步处理。
    *   暴露接口 `/generate`, `/encode`。

## 5. 实现步骤
1.  **设置**: 配置带有 LibTorch 的 CMake。
2.  **核心**: 定义 `Req`, `Batch` 结构体。
3.  **KVCache**: 实现 `RadixTreeNode` 和 `RadixCacheManager`。
4.  **模型**: 使用 LibTorch 实现最小化的 Llama。
5.  **调度器/引擎**: 实现调度循环和模型执行。
6.  **绑定/入口**: 一个 `main.cpp` 来运行简单的特定测试用例（例如 "Hello World"）。

## 6. 依赖项
*   **LibTorch (PyTorch C++ API)**: 用于张量操作和神经网络模块。
*   **brpc**: 提供高性能 RPC 和 HTTP 服务能力。
*   **gflags**: 用于命令行参数解析 (brpc 依赖)。
*   **glog**: 用于日志记录 (brpc 依赖)。
*   **protobuf**: 数据序列化 (brpc 依赖)。
*   **GTest**: 用于单元测试。
*   **nlohmann/json** (可选): 用于配置解析。

## 7. 优化考虑
*   **C++23 特性**: 使用 `std::expected` 进行错误处理，`std::print` 进行简单输出（在 glog 过于繁重时），以及 `std::span` 用于视图语义。
*   **内存对齐**: 确保张量分配对齐以适应 CUDA。
*   **批处理**: 使用高效的 scatter/gather 操作。
*   **缓存**: 优化 `RadixTree` 查找（可能是扁平化结构或哈希）。
