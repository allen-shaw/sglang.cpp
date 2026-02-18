#include "sglang/utils/tensor_utils.h"
#include <iostream>

namespace sglang {

void print_tensor(const torch::Tensor& tensor, const std::string& name) {
  std::cout << "Tensor[" << name << "]: " 
            << "Sizes=" << tensor.sizes() 
            << ", Dtype=" << tensor.dtype() 
            << ", Device=" << tensor.device() 
            << std::endl;
  // Optional: print content if small
  if (tensor.numel() <= 20) {
      std::cout << tensor << std::endl;
  }
}

torch::Tensor to_gpu(const torch::Tensor& tensor, int device_id) {
  // if (torch::cuda::is_available()) {
  //   return tensor.to(torch::Device(torch::kCUDA, device_id));
  // } else {
    std::cerr << "Warning: CUDA not available (compiled with CPU-only LibTorch), returning tensor on CPU" << std::endl;
    return tensor;
  // }
}

torch::Tensor move_to_cpu(const torch::Tensor& tensor) {
  return tensor.to(torch::kCPU);
}

} // namespace sglang
