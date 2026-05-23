#pragma once

#include <torch/torch.h>
#include <tuple>
#include <utility>
#include "sglang/models/config.h"
#include "sglang/models/utils.h"

namespace sglang {

class Qwen3MoEDecoderLayer : public torch::nn::Module {
 public:
    Qwen3MoEDecoderLayer(const ModelConfig& config, int layer_id);
    std::pair<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x, const torch::Tensor& positions, torch::Tensor residual);

 private:
    std::shared_ptr<RopeAttn> self_attn_;
    std::shared_ptr<MoEMLP> mlp_;
    std::shared_ptr<RMSNorm> input_layernorm_;
    std::shared_ptr<RMSNorm> post_attention_layernorm_;
};

class Qwen3MoEModel : public torch::nn::Module {
 public:
    explicit Qwen3MoEModel(const ModelConfig& config);
    torch::Tensor forward(const torch::Tensor& input_ids, const torch::Tensor& positions);

    std::shared_ptr<torch::nn::EmbeddingImpl> embed_tokens_;
    std::vector<std::shared_ptr<Qwen3MoEDecoderLayer>> layers_;
    std::shared_ptr<RMSNorm> norm_;
};

class Qwen3MoEForCausalLM : public torch::nn::Module {
 public:
    explicit Qwen3MoEForCausalLM(const ModelConfig& config);
    torch::Tensor forward(const torch::Tensor& input_ids, const torch::Tensor& positions);

    std::shared_ptr<Qwen3MoEModel> model_;
    std::shared_ptr<LinearReplicated> lm_head_;
};

} // namespace sglang
