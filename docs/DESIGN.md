# SGLang C++ Rewrite Design Document

## 1. Goal
Rewrite the [mini-sglang](https://github.com/sgl-project/mini-sglang) Python project in C++ (C++23 standard) to achieve better performance and lower latency, utilizing LibTorch for tensor operations.

## 2. Architecture Overview
The C++ implementation will closely mirror the Python architecture but adapted for C++ idioms (static typing, RAII, smart pointers). We will use the Google C++ Style Guide and technology stack (gtest, gflags, glog).

### Core Components
1.  **Engine** (`sglang::Engine`): The central controller, managing the model, KV cache, and device resources.
2.  **Scheduler** (`sglang::Scheduler`): Handles request scheduling, batching, and coordination between prefill and decode phases.
3.  **KVCache** (`sglang::kvcache`): Manages GPU memory for Key-Value cache blocks.
4.  **Model** (`sglang::models`): LibTorch-based model implementation (e.g., Llama, Qwen).
5.  **Context** (`sglang::global_context`): Thread-local or global context.
6.  **Server** (`sglang::server`): HTTP server using Cinatra.
7.  **Layers** (`sglang::layers`): Basic building blocks (RoPE, Attention, Linear).
8.  **Kernels** (`sglang::kernels`): Custom CUDA kernels (e.g., Radix matching).
9.  **Tokenizer** (`sglang::tokenizer`): Text encoding/decoding.
10. **Distributed** (`sglang::distributed`): Multi-GPU support (placeholder/future).
11. **Messages** (`sglang::messages`): Data structures for API communication, defined using standard C++ structs and serialized via JSON.
12. **MoE** (`sglang::moe`): Mixture-of-Experts implementation.
13. **LLM** (`sglang::llm`): High-level API for offline inference.

## 3. Directory Structure
We will use a standard C++ project structure within the `sglang.cpp` root.

```text
sglang.cpp/
    ├── CMakeLists.txt

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

## 4. Dependencies
*   **LibTorch (PyTorch C++ API)**: For tensor operations and neural network modules.
*   **Cinatra**: High-performance, easy-to-use C++20 header-only HTTP framework.
*   **gflags**: For command-line argument parsing.
*   **glog**: For logging.
*   **GTest**: For unit testing.
*   **nlohmann/json**: For JSON serialization and configuration parsing.
