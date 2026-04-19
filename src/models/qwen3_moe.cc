#include "sglang/models/qwen3_moe.h"

namespace sglang {

Qwen3MoEDecoderLayer::Qwen3MoEDecoderLayer(const ModelConfig& config, int layer_id) {
    self_attn_ = register_module("self_attn", std::make_shared<RopeAttn>(config, layer_id, true, false));
    mlp_ = register_module("mlp", std::make_shared<MoEMLP>(config));
    input_layernorm_ = register_module(
        "input_layernorm", std::make_shared<RMSNorm>(config.hidden_size, config.rms_norm_eps));
    post_attention_layernorm_ = register_module(
        "post_attention_layernorm", std::make_shared<RMSNorm>(config.hidden_size, config.rms_norm_eps));
}

torch::Tensor Qwen3MoEDecoderLayer::forward(torch::Tensor x, const torch::Tensor& positions) {
    auto residual = x;
    auto x_norm = input_layernorm_->forward(x);
    auto attn_out = self_attn_->forward(x_norm, positions);
    x = residual + attn_out;

    residual = x;
    x_norm = post_attention_layernorm_->forward(x);
    auto mlp_out = mlp_->forward(x_norm);
    x = residual + mlp_out;

    return x;
}

Qwen3MoEModel::Qwen3MoEModel(const ModelConfig& config) {
    embed_tokens_ = register_module(
        "embed_tokens",
        std::make_shared<torch::nn::EmbeddingImpl>(
        torch::nn::EmbeddingOptions(config.vocab_size, config.hidden_size)
    ));

    layers_.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        auto layer = std::make_shared<Qwen3MoEDecoderLayer>(config, i);
        register_module("layers_" + std::to_string(i), layer);
        layers_.push_back(layer);
    }

    norm_ = register_module("norm", std::make_shared<RMSNorm>(config.hidden_size, config.rms_norm_eps));
}

torch::Tensor Qwen3MoEModel::forward(const torch::Tensor& input_ids, const torch::Tensor& positions) {
    auto x = embed_tokens_->forward(input_ids);
    for (auto& layer : layers_) {
        x = layer->forward(x, positions);
    }
    return norm_->forward(x);
}

Qwen3MoEForCausalLM::Qwen3MoEForCausalLM(const ModelConfig& config) {
    model_ = register_module("model", std::make_shared<Qwen3MoEModel>(config));
    lm_head_ = register_module(
        "lm_head",
        std::make_shared<LinearReplicated>(
        config.hidden_size,
        config.vocab_size,
        false
    ));

    if (config.tie_word_embeddings) {
        lm_head_->weight = model_->embed_tokens_->weight;
    }
}

torch::Tensor Qwen3MoEForCausalLM::forward(const torch::Tensor& input_ids, const torch::Tensor& positions) {
    auto hidden = model_->forward(input_ids, positions);
    return lm_head_->forward(select_lm_head_input(hidden));
}

} // namespace sglang
