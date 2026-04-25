#include "sglang/layers/normalization.h"
#include <flashinfer/norm.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace sglang {
namespace {

template <typename scalar_t>
__global__ void qk_rmsnorm_3d_strided_kernel(scalar_t* __restrict__ x,
                                             const scalar_t* __restrict__ weight,
                                             int64_t rows,
                                             int64_t heads,
                                             int64_t d,
                                             int64_t stride_token,
                                             int64_t stride_head,
                                             float eps) {
    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (job >= rows) {
        return;
    }

    const int64_t token = job / heads;
    const int64_t head = job - token * heads;
    scalar_t* row_ptr = x + token * stride_token + head * stride_head;

    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < d; i += warpSize) {
        const float val = static_cast<float>(row_ptr[i]);
        sum_sq += val * val;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    const float scale = rsqrtf(sum_sq / static_cast<float>(d) + eps);
    for (int64_t i = threadIdx.x; i < d; i += warpSize) {
        const float val = static_cast<float>(row_ptr[i]);
        const float w = static_cast<float>(weight[i]);
        row_ptr[i] = static_cast<scalar_t>(val * scale * w);
    }
}

template <typename scalar_t, uint32_t VecSize>
__global__ void qk_rmsnorm_3d_strided_vec_kernel(scalar_t* __restrict__ x,
                                                 const scalar_t* __restrict__ weight,
                                                 int64_t rows,
                                                 int64_t heads,
                                                 int64_t d,
                                                 int64_t stride_token,
                                                 int64_t stride_head,
                                                 float eps) {
    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (job >= rows) {
        return;
    }

    const int64_t token = job / heads;
    const int64_t head = job - token * heads;
    scalar_t* row_ptr = x + token * stride_token + head * stride_head;

    constexpr int warp_size = 32;
    float sum_sq = 0.0f;
    for (int64_t base = static_cast<int64_t>(threadIdx.x) * VecSize;
         base < d;
         base += static_cast<int64_t>(warp_size) * VecSize) {
        flashinfer::vec_t<scalar_t, VecSize> input_vec;
        input_vec.load(row_ptr + base);
#pragma unroll
        for (uint32_t i = 0; i < VecSize; ++i) {
            const float val = static_cast<float>(input_vec[i]);
            sum_sq += val * val;
        }
    }

    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    const float scale = rsqrtf(sum_sq / static_cast<float>(d) + eps);
    for (int64_t base = static_cast<int64_t>(threadIdx.x) * VecSize;
         base < d;
         base += static_cast<int64_t>(warp_size) * VecSize) {
        flashinfer::vec_t<scalar_t, VecSize> input_vec;
        flashinfer::vec_t<scalar_t, VecSize> weight_vec;
        flashinfer::vec_t<scalar_t, VecSize> output_vec;
        input_vec.load(row_ptr + base);
        weight_vec.load(weight + base);
#pragma unroll
        for (uint32_t i = 0; i < VecSize; ++i) {
            output_vec[i] = static_cast<scalar_t>(
                static_cast<float>(input_vec[i]) * scale * static_cast<float>(weight_vec[i]));
        }
        output_vec.store(row_ptr + base);
    }
}

}  // namespace

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

void RMSNorm::forward_inplace_3d_strided(torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "Strided RMSNorm expects rank-3 tensor");
    TORCH_CHECK(x.stride(2) == 1, "Last dimension must be contiguous");
    const auto d = x.size(2);
    TORCH_CHECK(weight.size(0) == d, "RMSNorm weight size must match last dimension");

    const int64_t tokens = x.size(0);
    const int64_t heads = x.size(1);
    const int64_t rows = tokens * heads;
    if (rows == 0) {
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int kWarpsPerBlock = 4;
    dim3 blocks((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    dim3 threads(32, kWarpsPerBlock);
    constexpr uint32_t kVecSize = 8;
    if (x.scalar_type() == torch::kFloat16) {
        if (d % kVecSize == 0) {
            qk_rmsnorm_3d_strided_vec_kernel<half, kVecSize><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<half*>(weight.data_ptr<at::Half>()),
                rows, heads, d, x.stride(0), x.stride(1), eps_);
        } else {
            qk_rmsnorm_3d_strided_kernel<half><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<half*>(weight.data_ptr<at::Half>()),
                rows, heads, d, x.stride(0), x.stride(1), eps_);
        }
    } else {
        if (d % kVecSize == 0) {
            qk_rmsnorm_3d_strided_vec_kernel<nv_bfloat16, kVecSize><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                reinterpret_cast<nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                rows, heads, d, x.stride(0), x.stride(1), eps_);
        } else {
            qk_rmsnorm_3d_strided_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                reinterpret_cast<nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                rows, heads, d, x.stride(0), x.stride(1), eps_);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void RMSNorm::fused_add_forward_inplace(torch::Tensor& x, torch::Tensor& residual) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda(), "x and residual must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2 && residual.dim() == 2,
                "FusedAddRMSNorm expects rank-2 tensors");
    TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual shapes must match");
    TORCH_CHECK(x.size(-1) > 0, "RMSNorm last dimension must be positive");
    const auto d = static_cast<uint32_t>(x.size(-1));
    TORCH_CHECK(weight.size(0) == static_cast<int64_t>(d),
                "RMSNorm weight size must match last dimension");
    const auto batch_size = static_cast<uint32_t>(x.numel() / x.size(-1));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (x.scalar_type() == torch::kFloat16) {
        auto status = flashinfer::norm::FusedAddRMSNorm<half>(
            reinterpret_cast<half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(residual.data_ptr<at::Half>()),
            reinterpret_cast<half*>(weight.data_ptr<at::Half>()),
            batch_size, d, eps_, stream
        );
        TORCH_CHECK(status == cudaSuccess, "FlashInfer FusedAddRMSNorm failed");
    } else {
        auto status = flashinfer::norm::FusedAddRMSNorm<nv_bfloat16>(
            reinterpret_cast<nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            batch_size, d, eps_, stream
        );
        TORCH_CHECK(status == cudaSuccess, "FlashInfer FusedAddRMSNorm failed");
    }
}

}  // namespace sglang
