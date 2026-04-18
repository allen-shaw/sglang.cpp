#pragma once

#include <memory>

#include <torch/torch.h>

#include "sglang/attention/backend.h"
#include "sglang/core/batch.h"
#include "sglang/kvcache/base.h"

namespace sglang {

// Forward declaration for MOE backend (to be defined later)
struct BaseMoeBackend;

struct Context {
    int page_size = 16;

    // Page table (always treats page_size = 1 internally)
    torch::Tensor page_table;
    
    // Backends
    std::shared_ptr<BaseAttnBackend> attn_backend;
    std::shared_ptr<BaseMoeBackend> moe_backend;

    // KV Cache pool
    std::shared_ptr<BaseKVCachePool> kv_cache;

    // Current active batch
    // Using simple pointer or shared_ptr. Python uses `_batch` and context manager.
    // In C++, we might use RAII guard or explicit set/clear.
    std::shared_ptr<Batch> batch;

    Context(int page_size, std::shared_ptr<BaseAttnBackend> attn_backend);

    std::shared_ptr<Batch> get_batch() const;
    void set_batch(std::shared_ptr<Batch> new_batch);
    void clear_batch();
};

// Global Context Management
void set_global_ctx(std::shared_ptr<Context> ctx);
void reset_global_ctx();
std::shared_ptr<Context> get_global_ctx();

// RAII Guard for setting batch in context (similar to @contextmanager forward_batch)
class BatchGuard {
public:
    BatchGuard(std::shared_ptr<Context> ctx, std::shared_ptr<Batch> batch);
    ~BatchGuard();

    // Disable copy/move
    BatchGuard(const BatchGuard&) = delete;
    BatchGuard& operator=(const BatchGuard&) = delete;

private:
    std::shared_ptr<Context> ctx_;
};

} // namespace sglang
