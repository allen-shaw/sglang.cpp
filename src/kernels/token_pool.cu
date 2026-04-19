#include "sglang/kernels/token_pool.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace sglang {
namespace {

__global__ void write_token_pool_kernel(int32_t* token_pool,
                                        int64_t rows,
                                        int64_t stride,
                                        int64_t cols,
                                        const int64_t* req_indices,
                                        const int64_t* positions,
                                        const int32_t* tokens,
                                        int64_t count) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const int64_t req_idx = req_indices[idx];
  const int64_t pos = positions[idx];
  if (req_idx < 0 || req_idx >= rows || pos < 0 || pos >= cols) {
    return;
  }
  token_pool[req_idx * stride + pos] = tokens[idx];
}

}  // namespace

void write_token_pool(const torch::Tensor& token_pool,
                      const torch::Tensor& req_indices,
                      const torch::Tensor& positions,
                      const torch::Tensor& tokens) {
  TORCH_CHECK(token_pool.is_cuda(), "token_pool must be a CUDA tensor");
  TORCH_CHECK(req_indices.is_cuda(), "req_indices must be a CUDA tensor");
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  TORCH_CHECK(tokens.is_cuda(), "tokens must be a CUDA tensor");
  TORCH_CHECK(token_pool.scalar_type() == torch::kInt32, "token_pool must be int32");
  TORCH_CHECK(req_indices.scalar_type() == torch::kInt64, "req_indices must be int64");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64, "positions must be int64");
  TORCH_CHECK(tokens.scalar_type() == torch::kInt32, "tokens must be int32");
  TORCH_CHECK(token_pool.dim() == 2, "token_pool must be a 2D tensor");
  TORCH_CHECK(req_indices.dim() == 1, "req_indices must be a 1D tensor");
  TORCH_CHECK(positions.dim() == 1, "positions must be a 1D tensor");
  TORCH_CHECK(tokens.dim() == 1, "tokens must be a 1D tensor");
  TORCH_CHECK(req_indices.numel() == positions.numel(),
              "req_indices and positions must have the same length");
  TORCH_CHECK(req_indices.numel() == tokens.numel(),
              "req_indices and tokens must have the same length");

  const int64_t count = tokens.numel();
  if (count == 0) {
    return;
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  write_token_pool_kernel<<<blocks, threads, 0, stream>>>(
      token_pool.data_ptr<int32_t>(),
      token_pool.size(0),
      token_pool.stride(0),
      token_pool.size(1),
      req_indices.data_ptr<int64_t>(),
      positions.data_ptr<int64_t>(),
      tokens.data_ptr<int32_t>(),
      count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sglang
