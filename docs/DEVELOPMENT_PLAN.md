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
*   **5.3 测试与验证 (`tests/engine`, `tests/scheduler`, `tests/models`)**
    *   **目标**: 在阶段 5 先建立稳定的功能回归，再逐步补充并发和资源压力测试。
    *   **基础功能测试**:
        *   `Engine` 单请求端到端: 从 prefill 开始，完成一轮 decode，验证请求状态推进与生成 token 数量。
        *   `Scheduler` 单请求端到端: 从 `submit()` 发起请求，完成一轮完整推理，验证 `DetokenizeMsg` 和 finished 语义。
        *   真实模型 `Qwen3` Prefill Forward: 输入 `"The capital of France is"`，验证最后一个位置的 next token 能生成 `Paris/巴黎`。
        *   真实模型 `Qwen3` Greedy Generation: 生成结果文本需包含 `Paris/巴黎`，用于兜住模型层与采样层的基本正确性。
        *   真实模型 `Scheduler` smoke test: 输入自然语言 prompt，例如 `"What is the capital of France?"`，打印生成结果并验证输出包含 `Paris/巴黎`。
    *   **当前优先补充的多请求/并发测试**:
        *   **第一阶段：Engine / Scheduler 维度**
            *   `Engine` 同批双请求推理: 两个请求在同一个 batch 中 prefill/decode，验证 page table、KV 写入和请求结束条件互不影响。
            *   `Engine` 错峰双请求推理: 第一个请求完成首轮 prefill 后，第二个请求再进入，再共同完成 decode。
            *   `Engine` 并发压力 sweep: 按 `8 / 16 / 64 / 128` 扫描请求并发度，直到显存不足或达到上限。
            *   `Scheduler` 两个请求同一时刻提交并全部完成。
            *   `Scheduler` 三个请求同一时刻提交并全部完成。
            *   `Scheduler` 并发压力 sweep: 按 `8 / 16 / 64 / 128` 扫描请求并发度，直到显存不足或达到上限。
            *   一个请求进入 decode 后，第二个请求再提交，验证错峰到达不会饿死。
            *   不同 `max_new_tokens` 的请求独立结束，短请求可先回收，长请求继续 decode。
            *   pending request abort。
            *   running decode request abort。
            *   prefill budget 不足时触发 chunked prefill，并最终完成。
            *   decode 中插入 chunked prefill，验证调度状态机正确。
            *   共享 prefix 的请求复用 radix cache，避免不必要的初始 chunking。
            *   顺序批次之间的 running slot / page table 资源可复用。
            *   真实模型双请求 smoke test，例如同时询问 France/Germany 首都，验证不同请求输出不串扰。
        *   **第二阶段：服务与接口维度**
            *   `LLM` 离线接口并发生成: 同一个 `LLM` 实例并发提交多个 prompt，验证 uid、输出文本和完成状态不串扰。
            *   `LLM` 并发压力 sweep: 按 `8 / 16 / 64 / 128` 扫描批量生成并发度，直到显存不足或达到上限。
            *   HTTP `/generate` 并发请求: 两个用户同时请求不同问题，验证响应互不污染。
            *   HTTP `/generate` 并发压力 sweep: 按 `8 / 16 / 64 / 128` 扫描客户端并发度，直到显存不足或达到上限。
            *   `ApiServer::handle_chat_completions` 并发调用: 结构化 `messages` 并发进入时，chat template 路径仍能稳定返回各自结果。
            *   后续可继续增加 streaming 并发场景，验证 SSE 分块顺序与 `[DONE]` 结束标记。
    *   **后续再补充的压力/一致性测试**:
        *   多个长 prompt 同时进入，制造 cache pressure，验证 eviction 后仍能推进。
        *   prefix cache 命中与 abort 混合场景。
        *   overlap scheduling 打开/关闭时结果一致性。
        *   更多真实模型多请求 case，例如 3 个以上 prompt 同时生成。
        *   CUDA graph 打开后的 batch 行为与关闭路径一致。

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
