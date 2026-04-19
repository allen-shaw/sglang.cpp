#include "sglang/models/utils.h"

#include "sglang/core/context.h"
#include "sglang/layers/activation.h"

namespace sglang {

torch::Tensor select_lm_head_input(const torch::Tensor& hidden_states) {
    auto ctx = get_global_ctx();
    if (!ctx || !ctx->batch || !ctx->batch->is_prefill() || !ctx->batch->attn_metadata) {
        return hidden_states;
    }

    auto indices = ctx->batch->attn_metadata->get_last_indices(ctx->batch->size());
    if (indices.scalar_type() != torch::kInt64) {
        indices = indices.to(torch::kInt64);
    }
    return hidden_states.index_select(0, indices).contiguous();
}

GatedMLP::GatedMLP(const ModelConfig& config) : hidden_act_(config.hidden_act) {
    gate_up_proj_ = register_module(
        "gate_up_proj",
        std::make_shared<LinearColParallelMerged>(
            config.hidden_size,
            std::vector<int>{config.intermediate_size, config.intermediate_size},
            false
        )
    );

    down_proj_ = register_module(
        "down_proj",
        std::make_shared<LinearRowParallel>(
            config.intermediate_size,
            config.hidden_size,
            false
        )
    );
}

torch::Tensor GatedMLP::forward(const torch::Tensor& x) {
    auto gate_up = gate_up_proj_->forward(x);

    auto dims = gate_up.sizes().vec();
    dims.back() /= 2;
    auto y = torch::empty(dims, gate_up.options());

    if (hidden_act_ == "silu") {
        silu_and_mul(y, gate_up);
    } else if (hidden_act_ == "gelu") {
        gelu_and_mul(y, gate_up);
    } else {
        TORCH_CHECK(false, "Unsupported activation function: ", hidden_act_);
    }

    return down_proj_->forward(y);
}

MoEMLP::MoEMLP(const ModelConfig& config) {
    gate_ = register_module(
        "gate",
        std::make_shared<LinearReplicated>(
            config.hidden_size,
            config.num_experts,
            false
        )
    );

    experts_ = register_module(
        "experts",
        std::make_shared<MoELayer>(
            config.num_experts,
            config.num_experts_per_tok,
            config.hidden_size,
            config.moe_intermediate_size,
            config.norm_topk_prob
        )
    );
}

torch::Tensor MoEMLP::forward(const torch::Tensor& hidden_states) {
    auto num_tokens = hidden_states.size(0);
    auto hidden_dim = hidden_states.size(1);

    auto reshaped_hs = hidden_states.view({-1, hidden_dim});
    auto router_logits = gate_->forward(reshaped_hs);

    auto final_hs = experts_->forward(reshaped_hs, router_logits);
    return final_hs.view({num_tokens, hidden_dim});
}

RopeAttn::RopeAttn(const ModelConfig& config, int layer_id, bool has_attn_bias, bool has_qk_norm)
    : has_qk_norm_(has_qk_norm) {
    qkv_proj_ = register_module(
        "qkv_proj",
        std::make_shared<LinearQKVMerged>(
            config.hidden_size,
            config.head_dim,
            config.num_qo_heads,
            config.num_kv_heads,
            has_attn_bias
        )
    );

    if (has_qk_norm_) {
        q_norm_ = register_module("q_norm", std::make_shared<RMSNorm>(config.head_dim, config.rms_norm_eps));
        k_norm_ = register_module("k_norm", std::make_shared<RMSNorm>(config.head_dim, config.rms_norm_eps));
    }

    rotary_ = std::make_shared<RotaryEmbedding>(
        config.rotary_config.head_dim,
        config.rotary_config.rotary_dim,
        config.rotary_config.max_position,
        config.rotary_config.base
    );

    attn_ = std::make_shared<AttentionLayer>(
        layer_id,
        config.num_qo_heads,
        config.num_kv_heads,
        config.head_dim,
        rotary_.get(),
        q_norm_.get(),
        k_norm_.get()
    );

    o_proj_ = register_module(
        "o_proj",
        std::make_shared<LinearOProj>(
            config.head_dim * config.num_qo_heads,
            config.hidden_size,
            false
        )
    );
}

torch::Tensor RopeAttn::forward(const torch::Tensor& x, const torch::Tensor& positions) {
    auto qkv = qkv_proj_->forward(x);
    auto o = attn_->forward(qkv, positions);
    return o_proj_->forward(o);
}

} // namespace sglang
