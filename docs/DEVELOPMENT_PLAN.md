# SGLang C++ 重写开发计划

基于对 `mini-sglang` 的分析和新的 C++ 架构设计，以下是推荐的开发顺序。该顺序优先考虑依赖最少的模块，以便进行增量测试。

## 第一阶段：核心基础设施 (Core Infrastructure)
**目标：** 构建基础数据结构和工具库，这是所有其他模块的基石。

*   **1.1 核心定义 (`src/core`, `src/utils`, `src/message`)**
    *   **文件**: `req.h`, `batch.h`, `context.h`, `tensor_utils.h`, `sampling_params.h`
    *   **参考 Python 文件**: `minisgl/core.py`, `minisgl/env.py`, `minisgl/utils/*.py`
    *   **任务**: 定义 `Req`, `Batch`, `SamplingParams` 等核心结构体，以及 API 消息格式（使用 C++ 结构体）。
    *   **依赖**: 无（仅标准库和 LibTorch 头文件）。
    *   **测试**: 单元测试验证序列化/反序列化和张量工具函数。

## 第二阶段：计算与内存基础 (Compute & Memory Foundation)
**目标：** 实现底层的计算内核和内存管理逻辑，特别是 Radix Attention。

*   **2.1 KV 缓存逻辑 (`src/kvcache`)**
    *   **文件**: `radix_tree.h`, `cache_manager.h`
    *   **参考 Python 文件**: `minisgl/kvcache/radix_manager.py`, `minisgl/kvcache/base.py`
    *   **任务**: 实现 `RadixTreeNode` 和 `RadixCacheManager`。
    *   **核心逻辑**: 前缀匹配 (`match_prefix`)、插入 (`insert_prefix`)、驱逐 (`evict`)。
    *   **依赖**: `core`, `LibTorch`。
    *   **测试**: 独立的单元测试 (GTest) 验证树操作，无需模型参与。
*   **2.2 自定义内核 (`src/kernels`, `src/attention`)**
    *   **文件**: `radix.h`, `backend.h`
    *   **参考 Python 文件**: `minisgl/kernel/radix.py`, `minisgl/attention/*.py`
    *   **任务**: 封装 CUDA 内核（如 Radix 匹配加速）和 Attention 后端接口。
    *   **依赖**: `kvcache`。

## 第三阶段：模型组件 (Model Components)
**目标：** 构建神经网络的基本模块。

*   **3.1 基础层 (`src/layers`)**
    *   **文件**: `rotary_embedding.h`, `activation.h`, `linear.h`
    *   **参考 Python 文件**: `minisgl/layers/*.py`
    *   **任务**: 实现 RoPE, Silu/Gelu 等基础算子。
    *   **依赖**: `LibTorch`, `kernels`。
*   **3.2 分词器 (`src/tokenizer`)**
    *   **文件**: `tokenizer.h`
    *   **参考 Python 文件**: `minisgl/tokenizer/*.py`
    *   **任务**: 封装 HuggingFace Tokenizers C++ 绑定或类似实现。
    *   **依赖**: 第三方分词库。

## 第四阶段：模型实现 (Model Implementation)
**目标：** 组装完整的神经网络模型。

*   **4.1 模型架构 (`src/models`, `src/moe`)**
    *   **文件**: `llama.h`, `moe.h`
    *   **参考 Python 文件**: `minisgl/models/*.py`, `minisgl/moe/*.py`
    *   **任务**: 使用 LibTorch `nn::Module` 实现 Llama 和 MoE 结构。
    *   **依赖**: `layers`, `attention`。
    *   **测试**: 加载权重并与 Python 版本进行逐层输出对比。

## 第五阶段：调度与执行 (Control Plane)
**目标：** 实现请求调度逻辑，管理批处理和执行流。

*   **5.1 调度管理器 (`src/scheduler`)**
    *   **文件**: `prefill.h`, `decode.h`, `scheduler.h`
    *   **参考 Python 文件**: `minisgl/scheduler/*.py`
    *   **任务**: 实现 `PrefillManager` (新请求), `DecodeManager` (运行中请求) 和主调度循环。
    *   **依赖**: `kvcache`, `core`, `attention`。
*   **5.2 引擎集成 (`src/engine`)**
    *   **文件**: `engine.h`
    *   **参考 Python 文件**: `minisgl/engine/*.py`
    *   **任务**: 实现 `Engine::forward_batch`，连接调度器和模型。
    *   **依赖**: `scheduler`, `models`.

## 第六阶段：服务与接口 (Interface Layer)
**目标：** 对外暴露服务接口。

*   **6.1 LLM 高级 API (`src/llm`)**
    *   **文件**: `llm.h`
    *   **参考 Python 文件**: `minisgl/llm/*.py`
    *   **任务**: 提供类似 Python `LLM` 类的简易离线推理接口。
    *   **依赖**: `engine`, `tokenizer`。
*   **6.2 HTTP 服务器 (`src/server`)**
    *   **文件**: `http_server.h`
    *   **参考 Python 文件**: `minisgl/server/*.py`
    *   **任务**: 使用 `Cinatra` 实现 `/generate` 等 API 接口。
    *   **依赖**: `engine`, `Cinatra`, `message`。

## 第七阶段：高级功能 (Advanced Features)
**目标：** 扩展功能。

*   **7.1 分布式支持 (`src/distributed`)**
    *   **文件**: `distributed.h`
    *   **参考 Python 文件**: `minisgl/distributed/*.py`
    *   **任务**: 实现多 GPU 通信 (NCCL)。
    *   **依赖**: `LibTorch Distributed`.

## 总结
建议从 **`src/core`** 和 **`src/kvcache`** 开始。这是最独立且包含系统核心逻辑（Radix Attention）的部分。
