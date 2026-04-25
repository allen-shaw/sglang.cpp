#pragma once
#include <torch/torch.h>

namespace sglang {

class RMSNorm : public torch::nn::Module {
 public:
    RMSNorm(int size, float eps);
    torch::Tensor forward(const torch::Tensor& x);
    void forward_inplace(torch::Tensor& x);
    void forward_inplace_3d_strided(torch::Tensor& x);
    void fused_add_forward_inplace(torch::Tensor& x, torch::Tensor& residual);
    torch::Tensor weight;
 private:
    float eps_;
};

}  // namespace sglang
