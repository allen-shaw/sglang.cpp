#include "sglang/models/qwen.h"

namespace sglang {

// --- Qwen2DecoderLayer ---
Qwen2DecoderLayer::Qwen2DecoderLayer(const ModelConfig& config, int layer_id) {
    // Qwen models use true for has_attn_bias
    self_attn_ = std::make_shared<RopeAttn>(config, layer_id, true, false);
    mlp_ = std::make_shared<GatedMLP>(config);
    input_layernorm_ = std::make_shared<RMSNorm>(config.hidden_size, config.rms_norm_eps);
    post_attention_layernorm_ = std::make_shared<RMSNorm>(config.hidden_size, config.rms_norm_eps);
}

torch::Tensor Qwen2DecoderLayer::forward(torch::Tensor x, const torch::Tensor& positions) {
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

// --- Qwen2Model ---
Qwen2Model::Qwen2Model(const ModelConfig& config) {
    embed_tokens_ = std::make_shared<torch::nn::EmbeddingImpl>(
        torch::nn::EmbeddingOptions(config.vocab_size, config.hidden_size)
    );
    embed_tokens_->weight = torch::empty({config.vocab_size, config.hidden_size});

    layers_.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.push_back(std::make_shared<Qwen2DecoderLayer>(config, i));
    }

    norm_ = std::make_shared<RMSNorm>(config.hidden_size, config.rms_norm_eps);
}

torch::Tensor Qwen2Model::forward(const torch::Tensor& input_ids, const torch::Tensor& positions) {
    auto x = embed_tokens_->forward(input_ids);
    for (auto& layer : layers_) {
        x = layer->forward(x, positions);
    }
    return norm_->forward(x);
}

// --- Qwen2ForCausalLM ---
Qwen2ForCausalLM::Qwen2ForCausalLM(const ModelConfig& config) {
    model_ = std::make_shared<Qwen2Model>(config);
    lm_head_ = std::make_shared<LinearReplicated>(
        config.hidden_size,
        config.vocab_size,
        false
    );

    if (config.tie_word_embeddings) {
        lm_head_->weight = model_->embed_tokens_->weight;
    }
}

torch::Tensor Qwen2ForCausalLM::forward(const torch::Tensor& input_ids, const torch::Tensor& positions) {
    auto hidden = model_->forward(input_ids, positions);
    return lm_head_->forward(hidden);
}

} // namespace sglang
