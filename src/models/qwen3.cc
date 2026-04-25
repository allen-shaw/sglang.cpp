#include "sglang/models/qwen3.h"

namespace sglang {

Qwen3DecoderLayer::Qwen3DecoderLayer(const ModelConfig& config, int layer_id) {
    self_attn_ = register_module(
        "self_attn",
        std::make_shared<RopeAttn>(config, layer_id, false, true)
    );
    mlp_ = register_module("mlp", std::make_shared<GatedMLP>(config));
    input_layernorm_ = register_module(
        "input_layernorm",
        std::make_shared<RMSNorm>(config.hidden_size, config.rms_norm_eps)
    );
    post_attention_layernorm_ = register_module(
        "post_attention_layernorm",
        std::make_shared<RMSNorm>(config.hidden_size, config.rms_norm_eps)
    );
}

std::pair<torch::Tensor, torch::Tensor> Qwen3DecoderLayer::forward(
    torch::Tensor x, const torch::Tensor& positions, torch::Tensor residual) {
    if (residual.defined()) {
        input_layernorm_->fused_add_forward_inplace(x, residual);
    } else {
        residual = x;
        x = input_layernorm_->forward(x);
    }

    auto attn_out = self_attn_->forward(x, positions);
    post_attention_layernorm_->fused_add_forward_inplace(attn_out, residual);

    x = mlp_->forward(attn_out);
    return {x, residual};
}

Qwen3Model::Qwen3Model(const ModelConfig& config) {
    embed_tokens_ = register_module(
        "embed_tokens",
        std::make_shared<torch::nn::EmbeddingImpl>(
            torch::nn::EmbeddingOptions(config.vocab_size, config.hidden_size)
        )
    );

    layers_.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        auto layer = std::make_shared<Qwen3DecoderLayer>(config, i);
        register_module("layers_" + std::to_string(i), layer);
        layers_.push_back(layer);
    }

    norm_ = register_module("norm", std::make_shared<RMSNorm>(config.hidden_size, config.rms_norm_eps));
}

torch::Tensor Qwen3Model::forward(const torch::Tensor& input_ids, const torch::Tensor& positions) {
    auto x = embed_tokens_->forward(input_ids);
    torch::Tensor residual;
    for (auto& layer : layers_) {
        std::tie(x, residual) = layer->forward(x, positions, residual);
    }
    if (residual.defined()) {
        norm_->fused_add_forward_inplace(x, residual);
        return x;
    }
    return norm_->forward(x);
}

Qwen3ForCausalLM::Qwen3ForCausalLM(const ModelConfig& config) {
    model_ = register_module("model", std::make_shared<Qwen3Model>(config));
    lm_head_ = register_module(
        "lm_head",
        std::make_shared<LinearReplicated>(
            config.hidden_size,
            config.vocab_size,
            false
        )
    );

    if (config.tie_word_embeddings) {
        lm_head_->weight = model_->embed_tokens_->weight;
    }
}

torch::Tensor Qwen3ForCausalLM::forward(const torch::Tensor& input_ids, const torch::Tensor& positions) {
    auto hidden = model_->forward(input_ids, positions);
    return lm_head_->forward(select_lm_head_input(hidden));
}

} // namespace sglang
