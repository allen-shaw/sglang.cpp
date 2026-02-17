#pragma once

#include <memory>
#include <stdexcept>

#include "sglang/core/batch.h"

namespace sglang {

// Forward declarations for backends
struct BaseAttnBackend;
struct BaseMoeBackend;

struct Context {
    int page_size = 16;
    
    // Backends
    std::shared_ptr<BaseAttnBackend> attn_backend;
    std::shared_ptr<BaseMoeBackend> moe_backend;

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
