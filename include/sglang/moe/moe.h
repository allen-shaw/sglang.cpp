#pragma once

#include <torch/torch.h>

namespace sglang {

class MoELayer : public torch::nn::Module {
 public:
    MoELayer(int num_experts, int top_k, int hidden_size, int intermediate_size, bool renormalize = false);
    torch::Tensor forward(const torch::Tensor& hidden_states, const torch::Tensor& router_logits);

    torch::Tensor w1;
    torch::Tensor w2;

 private:
    int num_experts_;
    int top_k_;
    int hidden_size_;
    int intermediate_size_;
    bool renormalize_;
};

} // namespace sglang
