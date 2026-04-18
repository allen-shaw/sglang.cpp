#include "sglang/layers/normalization.h"
#include <flashinfer/norm.cuh>
#include <ATen/cuda/CUDAContext.h>

namespace sglang {

RMSNorm::RMSNorm(int size, float eps) : eps_(eps) {
    weight = register_parameter(
        "weight",
        torch::ones({size}, torch::device(torch::kCUDA).dtype(torch::kFloat16))
    );
}

torch::Tensor RMSNorm::forward(const torch::Tensor& x) {
    auto out = torch::empty_like(x);
    out.copy_(x);
    forward_inplace(out);
    return out;
}

void RMSNorm::forward_inplace(torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() >= 2, "RMSNorm expects tensor rank >= 2");
    TORCH_CHECK(x.size(-1) > 0, "RMSNorm last dimension must be positive");
    const auto d = static_cast<uint32_t>(x.size(-1));
    TORCH_CHECK(weight.size(0) == static_cast<int64_t>(d),
                "RMSNorm weight size must match last dimension");
    const auto batch_size = static_cast<uint32_t>(x.numel() / x.size(-1));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (x.scalar_type() == torch::kFloat16) {
        flashinfer::norm::RMSNorm<half>(
            reinterpret_cast<half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<half*>(x.data_ptr<at::Half>()),
            batch_size, d, eps_, stream
        );
    } else {
        flashinfer::norm::RMSNorm<nv_bfloat16>(
            reinterpret_cast<nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            batch_size, d, eps_, stream
        );
    }
}

}  // namespace sglang
