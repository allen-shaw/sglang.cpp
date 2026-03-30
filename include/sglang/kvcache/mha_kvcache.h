#pragma once

#include <torch/torch.h>
#include "sglang/kvcache/base.h"

namespace sglang {

// Forward declaration
struct ModelConfig;

/// MHAKVCache — concrete KV cache pool using Multi-Head Attention layout.
/// Allocates a single buffer of shape [2, num_layers, num_pages, page_size, kv_heads, head_dim]
/// and provides k_cache/v_cache per layer, plus store_kv for scatter writes.
///
/// Matches Python: minisgl/kvcache/mha_pool.py
class MHAKVCache : public BaseKVCachePool {
 public:
    MHAKVCache(int num_kv_heads, int num_layers, int head_dim,
               int num_pages, int page_size,
               torch::Dtype dtype, torch::Device device);

    torch::Tensor k_cache(int layer_id) const override;
    torch::Tensor v_cache(int layer_id) const override;
    void store_kv(const torch::Tensor& key, const torch::Tensor& val,
                  const torch::Tensor& out_loc, int layer_id) override;

    torch::Device device() const override { return device_; }
    torch::Dtype dtype() const override { return kv_buffer_.scalar_type(); }
    int num_layers() const override { return num_layers_; }

 private:
    torch::Tensor kv_buffer_;   // [2, num_layers, num_pages * page_size, kv_heads, head_dim]
    torch::Tensor k_buffer_;    // kv_buffer_[0]
    torch::Tensor v_buffer_;    // kv_buffer_[1]
    torch::Device device_;
    int num_layers_;
    int num_pages_;
    int page_size_;
    int kv_heads_;
    int head_dim_;
};

}  // namespace sglang
