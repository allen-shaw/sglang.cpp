#pragma once
#include <torch/torch.h>

namespace sglang {

class RMSNorm {
 public:
    RMSNorm(int size, float eps);
    torch::Tensor forward(const torch::Tensor& x);
    void forward_inplace(torch::Tensor& x);

 private:
    float eps_;
    torch::Tensor weight_;
};

}  // namespace sglang
