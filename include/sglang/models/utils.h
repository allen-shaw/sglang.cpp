#pragma once

#include <torch/torch.h>
#include <memory>
#include "sglang/models/config.h"
#include "sglang/layers/linear.h"
#include "sglang/layers/attention_layer.h"
#include "sglang/layers/normalization.h"
// #include "sglang/moe/moe.h"  // Forward declare for now

namespace sglang {

class MoELayer; // Forward declaration

class GatedMLP : public torch::nn::Module {
 public:
    explicit GatedMLP(const ModelConfig& config);
    torch::Tensor forward(const torch::Tensor& x);

 private:
    std::shared_ptr<LinearColParallelMerged> gate_up_proj_;
    std::shared_ptr<LinearRowParallel> down_proj_;
    std::string hidden_act_;
};

class MoEMLP : public torch::nn::Module {
 public:
    explicit MoEMLP(const ModelConfig& config);
    torch::Tensor forward(const torch::Tensor& hidden_states);

 private:
    std::shared_ptr<LinearReplicated> gate_;
    std::shared_ptr<MoELayer> experts_;
};

class RopeAttn : public torch::nn::Module {
 public:
    RopeAttn(const ModelConfig& config, int layer_id, bool has_attn_bias = false, bool has_qk_norm = false);
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& positions);

 private:
    std::shared_ptr<LinearQKVMerged> qkv_proj_;
    std::shared_ptr<LinearOProj> o_proj_;
    std::shared_ptr<RMSNorm> q_norm_;
    std::shared_ptr<RMSNorm> k_norm_;
    std::shared_ptr<RotaryEmbedding> rotary_;
    std::shared_ptr<AttentionLayer> attn_;
    bool has_qk_norm_;
};

} // namespace sglang
