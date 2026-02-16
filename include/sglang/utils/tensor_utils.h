#pragma once

#include <torch/torch.h>
#include <optional>
#include <string>

namespace sglang {

// Utility functions for tensor operations
void print_tensor(const torch::Tensor& tensor, const std::string& name = "");

torch::Tensor to_gpu(const torch::Tensor& tensor, int device_id = 0);

torch::Tensor move_to_cpu(const torch::Tensor& tensor);

// Helper to create a tensor from a vector
template <typename T>
torch::Tensor vec_to_tensor(const std::vector<T>& vec, torch::Device device = torch::kCPU);

} // namespace sglang
