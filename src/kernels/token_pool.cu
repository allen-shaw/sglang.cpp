#include "sglang/kernels/token_pool.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace sglang {
namespace {

__global__ void write_token_pool_kernel(int32_t* token_pool,
                                        int64_t rows,
                                        int64_t stride,
                                        int64_t cols,
                                        const int32_t* req_indices,
                                        const int32_t* positions,
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

__global__ void gather_int32_2d_kernel(const int32_t* table,
                                       int64_t rows,
                                       int64_t stride,
                                       int64_t cols,
                                       const int32_t* req_indices,
                                       const int32_t* positions,
                                       int32_t* output,
                                       int64_t count) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const int64_t req_idx = req_indices[idx];
  const int64_t pos = positions[idx];
  output[idx] = (req_idx >= 0 && req_idx < rows && pos >= 0 && pos < cols)
                    ? table[req_idx * stride + pos]
                    : 0;
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
  TORCH_CHECK(req_indices.scalar_type() == torch::kInt32, "req_indices must be int32");
  TORCH_CHECK(positions.scalar_type() == torch::kInt32, "positions must be int32");
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
      req_indices.data_ptr<int32_t>(),
      positions.data_ptr<int32_t>(),
      tokens.data_ptr<int32_t>(),
      count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor gather_int32_2d(const torch::Tensor& table,
                              const torch::Tensor& req_indices,
                              const torch::Tensor& positions) {
  TORCH_CHECK(table.is_cuda(), "table must be a CUDA tensor");
  TORCH_CHECK(req_indices.is_cuda(), "req_indices must be a CUDA tensor");
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  TORCH_CHECK(table.scalar_type() == torch::kInt32, "table must be int32");
  TORCH_CHECK(req_indices.scalar_type() == torch::kInt32, "req_indices must be int32");
  TORCH_CHECK(positions.scalar_type() == torch::kInt32, "positions must be int32");
  TORCH_CHECK(table.dim() == 2, "table must be a 2D tensor");
  TORCH_CHECK(req_indices.dim() == 1, "req_indices must be a 1D tensor");
  TORCH_CHECK(positions.dim() == 1, "positions must be a 1D tensor");
  TORCH_CHECK(req_indices.numel() == positions.numel(),
              "req_indices and positions must have the same length");

  auto output = torch::empty(
      {req_indices.numel()},
      torch::TensorOptions().dtype(torch::kInt32).device(table.device()));
  const int64_t count = output.numel();
  if (count == 0) {
    return output;
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  gather_int32_2d_kernel<<<blocks, threads, 0, stream>>>(
      table.data_ptr<int32_t>(),
      table.size(0),
      table.stride(0),
      table.size(1),
      req_indices.data_ptr<int32_t>(),
      positions.data_ptr<int32_t>(),
      output.data_ptr<int32_t>(),
      count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace sglang
