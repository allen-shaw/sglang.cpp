#include "sglang/layers/normalization.h"
#include <flashinfer/norm.cuh>
#include <ATen/cuda/CUDAContext.h>

namespace sglang {

RMSNorm::RMSNorm(int size, float eps) : eps_(eps) {
    weight_ = torch::ones({size}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
}

torch::Tensor RMSNorm::forward(const torch::Tensor& x) {
    auto out = torch::empty_like(x);
    forward_inplace(out);
    return out;
}

void RMSNorm::forward_inplace(torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    uint32_t batch_size = x.size(0);
    uint32_t d = x.size(1);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (x.scalar_type() == torch::kFloat16) {
        flashinfer::norm::RMSNorm<half>(
            reinterpret_cast<half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(weight_.data_ptr<at::Half>()),
            reinterpret_cast<half*>(x.data_ptr<at::Half>()),
            batch_size, d, eps_, stream
        );
    } else {
        flashinfer::norm::RMSNorm<nv_bfloat16>(
            reinterpret_cast<nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<nv_bfloat16*>(weight_.data_ptr<at::BFloat16>()),
            reinterpret_cast<nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            batch_size, d, eps_, stream
        );
    }
}

}  // namespace sglang
