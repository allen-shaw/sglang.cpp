#include "sglang/layers/attention_layer.h"
#include "sglang/core/context.h"
#include "sglang/distributed/distributed.h"

namespace sglang {

AttentionLayer::AttentionLayer(int layer_id, int num_qo_heads, int num_kv_heads, int head_dim,
                               RotaryEmbedding* rotary, RMSNorm* q_norm, RMSNorm* k_norm)
    : layer_id_(layer_id), head_dim_(head_dim), rotary_(rotary), q_norm_(q_norm), k_norm_(k_norm) {
    num_qo_heads_ = divide_even(num_qo_heads, tp_size(), "num_qo_heads");
    num_kv_heads_ = divide_even(num_kv_heads, tp_size(), "num_kv_heads");
    
    qo_attn_dim_ = num_qo_heads_ * head_dim;
    kv_attn_dim_ = num_kv_heads_ * head_dim;
}

torch::Tensor AttentionLayer::forward(const torch::Tensor& qkv, const torch::Tensor& positions) {
    // qkv is expected to be [total_tokens, qo_attn_dim + 2 * kv_attn_dim]
    auto q = qkv.narrow(-1, 0, qo_attn_dim_);
    auto k = qkv.narrow(-1, qo_attn_dim_, kv_attn_dim_);
    auto v = qkv.narrow(-1, qo_attn_dim_ + kv_attn_dim_, kv_attn_dim_);

    auto q_view = q.view({-1, num_qo_heads_, head_dim_});
    auto k_view = k.view({-1, num_kv_heads_, head_dim_});

    // Apply QK norm if present (Qwen3 uses this)
    if (q_norm_ && k_norm_) {
        fused_qk_rmsnorm_inplace_3d_strided(q_view, k_view, *q_norm_, *k_norm_);
    } else if (q_norm_) {
        q_norm_->forward_inplace_3d_strided(q_view);
    } else if (k_norm_) {
        k_norm_->forward_inplace_3d_strided(k_view);
    }

    // Apply RoPE using precomputed cos/sin cache
    TORCH_CHECK(rotary_, "AttentionLayer::forward requires rotary embedding");
    auto positions_mut = positions;  // need non-const for forward_inplace
    rotary_->forward_inplace(positions_mut, q_view, k_view);

    // Execute attention via global context's backend
    // Python: o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
    auto ctx = get_global_ctx();
    auto batch = ctx->get_batch();
    TORCH_CHECK(batch, "AttentionLayer::forward requires an active batch in Context");
    TORCH_CHECK(ctx->attn_backend, "AttentionLayer::forward requires attn_backend in Context");

    auto o = ctx->attn_backend->forward(q_view, k, v, layer_id_, *batch);

    // Output shape: [total_tokens, qo_attn_dim]
    return o.view({-1, qo_attn_dim_});
}

}  // namespace sglang
