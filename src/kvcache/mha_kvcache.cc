#include "sglang/kvcache/mha_kvcache.h"

namespace sglang {

MHAKVCache::MHAKVCache(int num_kv_heads, int num_layers, int head_dim,
                       int num_pages, int page_size,
                       torch::Dtype dtype, torch::Device device)
    : device_(device), num_layers_(num_layers),
      num_pages_(num_pages), page_size_(page_size),
      kv_heads_(num_kv_heads), head_dim_(head_dim) {
    
    // Allocate: [2, num_layers, num_pages, page_size, kv_heads, head_dim]
    // We flatten num_pages * page_size into a single dim for storage
    int64_t total_slots = static_cast<int64_t>(num_pages) * page_size;
    kv_buffer_ = torch::empty(
        {2, num_layers, total_slots, num_kv_heads, head_dim},
        torch::TensorOptions().dtype(dtype).device(device)
    );
    k_buffer_ = kv_buffer_[0];  // [num_layers, total_slots, kv_heads, head_dim]
    v_buffer_ = kv_buffer_[1];
}

torch::Tensor MHAKVCache::k_cache(int layer_id) const {
    return k_buffer_[layer_id];  // [total_slots, kv_heads, head_dim]
}

torch::Tensor MHAKVCache::v_cache(int layer_id) const {
    return v_buffer_[layer_id];  // [total_slots, kv_heads, head_dim]
}

void MHAKVCache::store_kv(const torch::Tensor& key, const torch::Tensor& val,
                           const torch::Tensor& out_loc, int layer_id) {
    // key shape: [num_tokens, kv_heads, head_dim] (or [num_tokens, kv_attn_dim])
    // val shape: [num_tokens, kv_heads, head_dim] (or [num_tokens, kv_attn_dim])
    // out_loc shape: [num_tokens] — indices into the cache slots
    
    auto k_buf = k_buffer_[layer_id];  // [total_slots, kv_heads, head_dim]
    auto v_buf = v_buffer_[layer_id];

    // Reshape key/val to [num_tokens, kv_heads, head_dim] if needed
    auto k = key.dim() == 2 ? key.view({-1, kv_heads_, head_dim_}) : key;
    auto v = val.dim() == 2 ? val.view({-1, kv_heads_, head_dim_}) : val;
    
    // Use index_copy_ along dim 0 to scatter tokens into their cache slots
    k_buf.index_copy_(0, out_loc, k);
    v_buf.index_copy_(0, out_loc, v);
}

}  // namespace sglang
