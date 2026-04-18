#pragma once

#include <cstdint>

#include <torch/torch.h>

namespace sglang {

/// Compare two 1-D integer tensors element-by-element on CPU.
/// Returns the length of the longest matching prefix.
int64_t fast_compare_key(const torch::Tensor& lhs, const torch::Tensor& rhs);

}  // namespace sglang
