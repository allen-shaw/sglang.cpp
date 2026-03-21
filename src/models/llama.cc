#include "sglang/models/llama.h"

namespace sglang {

// --- LlamaDecoderLayer ---
LlamaDecoderLayer::LlamaDecoderLayer(const ModelConfig& config, int layer_id) {
    self_attn_ = std::make_shared<RopeAttn>(config, layer_id, false, false);
    mlp_ = std::make_shared<GatedMLP>(config);
    input_layernorm_ = std::make_shared<RMSNorm>(config.hidden_size, config.rms_norm_eps);
    post_attention_layernorm_ = std::make_shared<RMSNorm>(config.hidden_size, config.rms_norm_eps);
}

torch::Tensor LlamaDecoderLayer::forward(torch::Tensor x, const torch::Tensor& positions) {
    // Self Attention Block
    auto residual = x;
    auto x_norm = input_layernorm_->forward(x);
    auto attn_out = self_attn_->forward(x_norm, positions);
    x = residual + attn_out;

    // MLP Block
    residual = x;
    x_norm = post_attention_layernorm_->forward(x);
    auto mlp_out = mlp_->forward(x_norm);
    x = residual + mlp_out;

    return x;
}

// --- LlamaModel ---
LlamaModel::LlamaModel(const ModelConfig& config) {
    embed_tokens_ = std::make_shared<torch::nn::EmbeddingImpl>(
        torch::nn::EmbeddingOptions(config.vocab_size, config.hidden_size)
    );
    embed_tokens_->weight = torch::empty({config.vocab_size, config.hidden_size});

    layers_.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.push_back(std::make_shared<LlamaDecoderLayer>(config, i));
    }

    norm_ = std::make_shared<RMSNorm>(config.hidden_size, config.rms_norm_eps);
}

torch::Tensor LlamaModel::forward(const torch::Tensor& input_ids, const torch::Tensor& positions) {
    auto x = embed_tokens_->forward(input_ids);
    for (auto& layer : layers_) {
        x = layer->forward(x, positions);
    }
    return norm_->forward(x);
}

// --- LlamaForCausalLM ---
LlamaForCausalLM::LlamaForCausalLM(const ModelConfig& config) {
    model_ = std::make_shared<LlamaModel>(config);
    lm_head_ = std::make_shared<LinearReplicated>(
        config.hidden_size,
        config.vocab_size,
        false
    );

    if (config.tie_word_embeddings) {
        // Tie weights: Replicate the base pointer
        lm_head_->weight = model_->embed_tokens_->weight;
    }
}

torch::Tensor LlamaForCausalLM::forward(const torch::Tensor& input_ids, const torch::Tensor& positions) {
    auto hidden = model_->forward(input_ids, positions);
    return lm_head_->forward(hidden);
}

} // namespace sglang
