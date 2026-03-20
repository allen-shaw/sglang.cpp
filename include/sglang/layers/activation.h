#pragma once
#include <torch/torch.h>

namespace sglang {

/// Fused SiLU and Multiplication: out = SiLU(x) * y
/// where x is the first half of the input tensor, and y is the second half.
void silu_and_mul(torch::Tensor& out, const torch::Tensor& input);

/// Fused GeLU and Multiplication: out = GeLU(x) * y
void gelu_and_mul(torch::Tensor& out, const torch::Tensor& input);

}  // namespace sglang
