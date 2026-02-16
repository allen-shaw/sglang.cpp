# Mini-SGLang C++ Rewrite Design Document

## 1. Goal
Rewrite the [mini-sglang](https://github.com/sgl-project/mini-sglang) Python project in C++ (C++23 standard) to achieve better performance and lower latency, utilizing LibTorch for tensor operations.

## 2. Architecture Overview
The C++ implementation will closely mirror the Python architecture but adapted for C++ idioms (static typing, RAII, smart pointers). We will use the Google C++ Style Guide and technology stack (gtest, gflags, glog).

### Core Components
1.  **Engine** (`sglang::Engine`): The central controller, managing the model, KV cache, and device resources.
2.  **Scheduler** (`sglang::Scheduler`): Handles request scheduling, batching, and coordination between prefill and decode phases.
3.  **KVCache** (`sglang::kvcache`): Manages GPU memory for Key-Value cache blocks.
    *   `RadixCacheManager` & `RadixTreeNode`: Implements prefix caching using a Radix Tree.
    *   `CacheManager`: Abstract interface for memory allocation.
4.  **Model** (`sglang::models`): LibTorch-based model implementation (e.g., Llama, Qwen).
5.  **Context** (`sglang::global_context`): Thread-local or global context for managing request state during execution.
6.  **Server** (`sglang::server`): Uses **brpc** to handle HTTP/RPC requests, replacing Python's FastAPI.

## 3. Directory Structure
We will use a standard C++ project structure within the `sglang.cpp` root.

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
│       │   └── engine.h
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
│       └── utils/
│           └── tensor_utils.h
├── src/
│   ├── core/
│   ├── engine/
│   ├── kvcache/
│   ├── models/
│   ├── scheduler/
│   └── main.cpp
└── tests/
```

## 4. Detailed Component Design

### 4.1 Core Data Structures (`include/sglang/core/`)
*   **Req**: Struct holding request state (`input_ids`, `output_ids`, `sampling_params`, `cache_handle`, `uid`).
    *   Use `std::shared_ptr<Req>` for lifetime management.
*   **Batch**: Struct holding a vector of `Req`s and phase information (`PREFILL`, `DECODE`).

### 4.2 KV Cache (`include/sglang/kvcache/`)
*   **RadixTreeNode**:
    *   Represents a node in the prefix tree.
    *   Holds `key` (tokens) and `value` (indices in KV cache).
    *   Uses `std::shared_mutex` for thread-safe access if needed (single-scheduler assumption initially).
*   **RadixCacheManager**:
    *   Manages the Radix Tree.
    *   Implements `match_prefix`, `insert_prefix`, `evict`.

### 4.3 Scheduler (`include/sglang/scheduler/`)
*   **Scheduler**:
    *   Owns `Engine`, `CacheManager`.
    *   `run_loop()`: Main loop for processing requests via polling or condition variables.
    *   `schedule_next_batch()`: Selects requests from waiting queue.
*   **PrefillManager**: Handles new requests.
*   **DecodeManager**: Handles running requests.

### 4.4 Engine (`include/sglang/engine/`)
*   **Engine**:
    *   Initializes `torch::Device`.
    *   Loads model weights (using `torch::load` or safetensors adapter).
    *   `forward_batch(Batch& batch)`: Executes the model using `torch::jit` or custom `nn::Module`.

### 4.5 Server (`include/sglang/server/`)
*   **HttpServer**:
    *   Uses **brpc** (Baidu RPC, widely used C++ framework compatible with gflags/glog/protobuf) to handle HTTP/REST requests.
    *   Replaces FastAPI's async handling with brpc's high-performance event loop (bthread).
    *   Exposes endpoints `/generate`, `/encode`.

## 5. Implementation Steps
1.  **Setup**: Configure CMake with LibTorch.
2.  **Core**: Define `Req`, `Batch` structs.
3.  **KVCache**: Implement `RadixTreeNode` and `RadixCacheManager`.
4.  **Model**: Minimal Llama implementation using LibTorch.
5.  **Scheduler/Engine**: Implement the scheduling loop and model execution.
6.  **Binding/Entry**: A `main.cpp` to run a simple specific test case (e.g., "Hello World").

## 6. Dependencies
*   **LibTorch (PyTorch C++ API)**: For tensor operations and neural network modules.
*   **brpc**: Provide high-performance RPC and HTTP server capabilities.
*   **gflags**: For command-line argument parsing (required by brpc).
*   **glog**: For logging (required by brpc).
*   **protobuf**: For data serialization (required by brpc).
*   **GTest**: For unit testing.
*   **nlohmann/json** (Optional): For configuration parsing.

## 7. Optimization Considerations
*   **C++23 Features**: Use `std::expected` for error handling, `std::print` for simple output (where glog is too much), and `std::span` for view semantics.
*   **Memory alignment**: Ensure tensor allocations are aligned for CUDA.
*   **Batching**: Use efficient scatter/gather operations.
*   **Caching**: Optimize `RadixTree` lookup (maybe flattened structure or hashing).
