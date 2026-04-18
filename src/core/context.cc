#include "sglang/core/context.h"
#include <iostream>

namespace sglang {

// Global context pointer
static std::shared_ptr<Context> g_global_ctx = nullptr;

Context::Context(int page_size, std::shared_ptr<BaseAttnBackend> attn_backend)
    : page_size(page_size), attn_backend(attn_backend) {
}

std::shared_ptr<Batch> Context::get_batch() const {
  if (!batch) {
    // throw std::runtime_error("No active batch in context");
    // Or return nullptr? Python raises assertion error.
    // For now, let's return nullptr to allow checking.
    // Ideally should be checked before access.
    return nullptr;
  }
  return batch;
}

void Context::set_batch(std::shared_ptr<Batch> new_batch) {
  if (batch) {
    throw std::runtime_error("Nested forward_batch is not allowed / Batch already set");
  }
  batch = new_batch;
}

void Context::clear_batch() {
  batch = nullptr;
}

void set_global_ctx(std::shared_ptr<Context> ctx) {
  g_global_ctx = ctx;
}

void reset_global_ctx() {
  g_global_ctx.reset();
}

std::shared_ptr<Context> get_global_ctx() {
  if (!g_global_ctx) {
    throw std::runtime_error("Global context is not set");
  }
  return g_global_ctx;
}

// BatchGuard Implementation
BatchGuard::BatchGuard(std::shared_ptr<Context> ctx, std::shared_ptr<Batch> batch)
    : ctx_(ctx) {
  ctx_->set_batch(batch);
}

BatchGuard::~BatchGuard() {
  ctx_->clear_batch();
}

} // namespace sglang
