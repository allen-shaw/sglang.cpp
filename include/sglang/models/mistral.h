#pragma once

#include <torch/torch.h>
#include "sglang/models/config.h"
#include "sglang/models/utils.h"

namespace sglang {

class MistralDecoderLayer : public torch::nn::Module {
 public:
    MistralDecoderLayer(const ModelConfig& config, int layer_id);
    torch::Tensor forward(torch::Tensor x, const torch::Tensor& positions);

 private:
    std::shared_ptr<RopeAttn> self_attn_;
    std::shared_ptr<GatedMLP> mlp_;
    std::shared_ptr<RMSNorm> input_layernorm_;
    std::shared_ptr<RMSNorm> post_attention_layernorm_;
};

class MistralModel : public torch::nn::Module {
 public:
    explicit MistralModel(const ModelConfig& config);
    torch::Tensor forward(const torch::Tensor& input_ids, const torch::Tensor& positions);

    std::shared_ptr<torch::nn::EmbeddingImpl> embed_tokens_;
    std::vector<std::shared_ptr<MistralDecoderLayer>> layers_;
    std::shared_ptr<RMSNorm> norm_;
};

class MistralForCausalLM : public torch::nn::Module {
 public:
    explicit MistralForCausalLM(const ModelConfig& config);
    torch::Tensor forward(const torch::Tensor& input_ids, const torch::Tensor& positions);

    std::shared_ptr<MistralModel> model_;
    std::shared_ptr<LinearReplicated> lm_head_;
};

} // namespace sglang
