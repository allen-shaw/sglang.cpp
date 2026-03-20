#include "sglang/layers/attention_layer.h"

namespace sglang {

AttentionLayer::AttentionLayer(int layer_id, int num_qo_heads, int num_kv_heads, int head_dim,
                               RotaryEmbedding* rotary, RMSNorm* q_norm, RMSNorm* k_norm)
    : layer_id_(layer_id), head_dim_(head_dim), rotary_(rotary), q_norm_(q_norm), k_norm_(k_norm) {
    
    int tp_size = 1; // FIXME: distributed
    num_qo_heads_ = num_qo_heads / tp_size;
    num_kv_heads_ = num_kv_heads / tp_size; // FIXME: allow replicate
    
    qo_attn_dim_ = num_qo_heads_ * head_dim;
    kv_attn_dim_ = num_kv_heads_ * head_dim;
}

torch::Tensor AttentionLayer::forward(const torch::Tensor& qkv, const torch::Tensor& positions) {
    // qkv is expected to be [total_tokens, qo_attn_dim + 2 * kv_attn_dim]
    auto splits = qkv.split({qo_attn_dim_, kv_attn_dim_, kv_attn_dim_}, -1);
    auto q = splits[0];
    auto k = splits[1];
    auto v = splits[2];

    if (q_norm_) {
        // q_norm needs shape to be 2D, e.g. [..., head_dim] but it's applied on full heads.
        q_norm_->forward_inplace(q);
    }
    if (k_norm_) {
        k_norm_->forward_inplace(k);
    }

    auto q_view = q.view({-1, num_qo_heads_, head_dim_});
    auto k_view = k.view({-1, num_kv_heads_, head_dim_});

    // We assume the cos_cache and sin_cache are passed globally or fetched via context,
    // but the python implementation handles it via get_global_ctx().
    // For now we assert false as it needs further wiring with Batch context.
    TORCH_CHECK(false, "AttentionLayer::forward requires cos_cache and sin_cache wiring, which will be integrated in Phase 4");

    return torch::Tensor();
}

}  // namespace sglang
