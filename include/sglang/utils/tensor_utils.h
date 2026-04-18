#pragma once

#include <torch/torch.h>
#include <optional>
#include <string>
#include <vector>

namespace sglang {

// Utility functions for tensor operations
void print_tensor(const torch::Tensor& tensor, const std::string& name = "");

torch::Tensor to_gpu(const torch::Tensor& tensor, int device_id = 0);

torch::Tensor move_to_cpu(const torch::Tensor& tensor);

// Helper to create a tensor from a vector
template <typename T>
torch::Tensor vec_to_tensor(const std::vector<T>& vec, torch::Device device = torch::kCPU) {
    auto options = torch::TensorOptions().device(device);
    // torch::tensor infers dtype from the data
    return torch::tensor(vec, options);
}

// RAII guard for temporarily changing the default torch dtype.
// Matches Python's torch_dtype context manager.
class TorchDtypeGuard {
 public:
  explicit TorchDtypeGuard(torch::Dtype dtype)
      : old_dtype_(torch::get_default_dtype()) {
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(dtype));
  }
  ~TorchDtypeGuard() {
    torch::set_default_dtype(old_dtype_);
  }

  TorchDtypeGuard(const TorchDtypeGuard&) = delete;
  TorchDtypeGuard& operator=(const TorchDtypeGuard&) = delete;

 private:
  caffe2::TypeMeta old_dtype_;
};

} // namespace sglang
