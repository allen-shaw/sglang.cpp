#include "sglang/layers/activation.h"
#include <flashinfer/activation.cuh>
#include <ATen/cuda/CUDAContext.h>

namespace sglang {

__device__ __forceinline__ float d_silu(const float& x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float d_gelu(const float& x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

void silu_and_mul(torch::Tensor& out, const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && out.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16, "Inputs must be FP16/BF16");

    int64_t num_tokens = input.size(0);
    int64_t d = input.size(1) / 2;

    int num_blocks = num_tokens;
    int num_threads = std::min(1024L, d);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (input.scalar_type() == torch::kFloat16) {
        auto kernel = flashinfer::activation::act_and_mul_kernel<half, d_silu>;
        kernel<<<num_blocks, num_threads, 0, stream>>>(
            reinterpret_cast<half*>(out.data_ptr<at::Half>()), 
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()), 
            d
        );
    } else {
        auto kernel = flashinfer::activation::act_and_mul_kernel<nv_bfloat16, d_silu>;
        kernel<<<num_blocks, num_threads, 0, stream>>>(
            reinterpret_cast<nv_bfloat16*>(out.data_ptr<at::BFloat16>()), 
            reinterpret_cast<const nv_bfloat16*>(input.data_ptr<at::BFloat16>()), 
            d
        );
    }
}

void gelu_and_mul(torch::Tensor& out, const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && out.is_cuda(), "Inputs must be CUDA tensors");
    int64_t num_tokens = input.size(0);
    int64_t d = input.size(1) / 2;

    int num_blocks = num_tokens;
    int num_threads = std::min(1024L, d);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (input.scalar_type() == torch::kFloat16) {
        auto kernel = flashinfer::activation::act_and_mul_kernel<half, d_gelu>;
        kernel<<<num_blocks, num_threads, 0, stream>>>(
            reinterpret_cast<half*>(out.data_ptr<at::Half>()), 
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()), 
            d
        );
    } else {
        auto kernel = flashinfer::activation::act_and_mul_kernel<nv_bfloat16, d_gelu>;
        kernel<<<num_blocks, num_threads, 0, stream>>>(
            reinterpret_cast<nv_bfloat16*>(out.data_ptr<at::BFloat16>()), 
            reinterpret_cast<const nv_bfloat16*>(input.data_ptr<at::BFloat16>()), 
            d
        );
    }
}

}  // namespace sglang
