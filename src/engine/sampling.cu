#include "sglang/engine/sampling.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <flashinfer/sampling.cuh>

namespace sglang {
namespace {

constexpr int64_t kProbDims = 2;
constexpr int64_t kUniformDimsNoFilter = 1;
constexpr int64_t kUniformDimsFilter = 2;

void check_probs(const torch::Tensor& probs) {
    TORCH_CHECK(probs.defined(), "probs must be defined");
    TORCH_CHECK(probs.is_cuda(), "FlashInfer sampling requires CUDA probs");
    TORCH_CHECK(probs.is_contiguous(), "FlashInfer sampling requires contiguous probs");
    TORCH_CHECK(probs.scalar_type() == torch::kFloat32,
                "FlashInfer sampling currently expects float32 probs");
    TORCH_CHECK(probs.dim() == kProbDims, "probs must have shape [batch, vocab]");
}

void check_uniform(const torch::Tensor& probs,
                   const torch::Tensor& uniform_samples,
                   int64_t expected_dims) {
    TORCH_CHECK(uniform_samples.defined(), "uniform_samples must be defined");
    TORCH_CHECK(uniform_samples.is_cuda(), "FlashInfer sampling requires CUDA uniform_samples");
    TORCH_CHECK(uniform_samples.is_contiguous(),
                "FlashInfer sampling requires contiguous uniform_samples");
    TORCH_CHECK(uniform_samples.scalar_type() == torch::kFloat32,
                "FlashInfer sampling currently expects float32 uniform_samples");
    TORCH_CHECK(uniform_samples.device() == probs.device(),
                "uniform_samples must be on the same device as probs");
    TORCH_CHECK(uniform_samples.dim() == expected_dims,
                "uniform_samples has unexpected dimensions");
    if (expected_dims == kUniformDimsNoFilter) {
        TORCH_CHECK(uniform_samples.size(0) == probs.size(0),
                    "uniform_samples must have shape [batch]");
    } else {
        TORCH_CHECK(uniform_samples.size(1) == probs.size(0),
                    "uniform_samples must have shape [max_rounds, batch]");
    }
}

torch::Tensor make_samples(const torch::Tensor& probs) {
    return torch::empty({probs.size(0)},
                        torch::TensorOptions().dtype(torch::kInt32).device(probs.device()));
}

torch::Tensor make_success(const torch::Tensor& probs) {
    return torch::empty({probs.size(0)},
                        torch::TensorOptions().dtype(torch::kBool).device(probs.device()));
}

void check_status(cudaError_t status, const char* name) {
    TORCH_CHECK(status == cudaSuccess,
                name,
                " failed with error: ",
                cudaGetErrorString(status));
}

}  // namespace

torch::Tensor flashinfer_sample_from_probs(const torch::Tensor& probs,
                                           const torch::Tensor& uniform_samples,
                                           bool deterministic) {
    check_probs(probs);
    check_uniform(probs, uniform_samples, kUniformDimsNoFilter);

    c10::cuda::CUDAGuard guard(probs.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(probs.device().index()).stream();
    auto samples = make_samples(probs);
    auto status = flashinfer::sampling::SamplingFromProb<float, int>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<float*>(uniform_samples.data_ptr()),
        static_cast<int*>(samples.data_ptr()),
        static_cast<uint32_t>(probs.size(0)),
        static_cast<uint32_t>(probs.size(1)),
        deterministic,
        stream);
    check_status(status, "FlashInfer SamplingFromProb");
    return samples;
}

FlashInferSamplingResult flashinfer_top_k_sample_from_probs(const torch::Tensor& probs,
                                                            const torch::Tensor& uniform_samples,
                                                            const torch::Tensor& top_k,
                                                            int64_t top_k_value,
                                                            bool deterministic) {
    check_probs(probs);
    check_uniform(probs, uniform_samples, kUniformDimsFilter);
    TORCH_CHECK(!top_k.defined() ||
                    (top_k.is_cuda() && top_k.is_contiguous() &&
                     top_k.scalar_type() == torch::kFloat32 &&
                     top_k.device() == probs.device() && top_k.numel() == probs.size(0)),
                "top_k must be a contiguous CUDA float32 tensor with shape [batch]");

    c10::cuda::CUDAGuard guard(probs.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(probs.device().index()).stream();
    auto samples = make_samples(probs);
    auto success = make_success(probs);
    auto status = flashinfer::sampling::TopKSamplingFromProb<float, int>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<float*>(uniform_samples.data_ptr()),
        static_cast<int*>(samples.data_ptr()),
        static_cast<bool*>(success.data_ptr()),
        top_k.defined() ? static_cast<float*>(top_k.data_ptr()) : nullptr,
        static_cast<uint32_t>(probs.size(0)),
        static_cast<uint32_t>(top_k_value),
        static_cast<uint32_t>(probs.size(1)),
        static_cast<uint32_t>(uniform_samples.size(0)),
        deterministic,
        stream);
    check_status(status, "FlashInfer TopKSamplingFromProb");
    return {samples, success};
}

FlashInferSamplingResult flashinfer_top_p_sample_from_probs(const torch::Tensor& probs,
                                                            const torch::Tensor& uniform_samples,
                                                            const torch::Tensor& top_p,
                                                            double top_p_value,
                                                            bool deterministic) {
    check_probs(probs);
    check_uniform(probs, uniform_samples, kUniformDimsFilter);
    TORCH_CHECK(!top_p.defined() ||
                    (top_p.is_cuda() && top_p.is_contiguous() &&
                     top_p.scalar_type() == torch::kFloat32 &&
                     top_p.device() == probs.device() && top_p.numel() == probs.size(0)),
                "top_p must be a contiguous CUDA float32 tensor with shape [batch]");

    c10::cuda::CUDAGuard guard(probs.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(probs.device().index()).stream();
    auto samples = make_samples(probs);
    auto success = make_success(probs);
    auto status = flashinfer::sampling::TopPSamplingFromProb<float, int>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<float*>(uniform_samples.data_ptr()),
        static_cast<int*>(samples.data_ptr()),
        static_cast<bool*>(success.data_ptr()),
        top_p.defined() ? static_cast<float*>(top_p.data_ptr()) : nullptr,
        static_cast<uint32_t>(probs.size(0)),
        static_cast<float>(top_p_value),
        static_cast<uint32_t>(probs.size(1)),
        static_cast<uint32_t>(uniform_samples.size(0)),
        deterministic,
        stream);
    check_status(status, "FlashInfer TopPSamplingFromProb");
    return {samples, success};
}

FlashInferSamplingResult flashinfer_top_k_top_p_sample_from_probs(const torch::Tensor& probs,
                                                                  const torch::Tensor& uniform_samples,
                                                                  const torch::Tensor& top_k,
                                                                  double top_k_value,
                                                                  const torch::Tensor& top_p,
                                                                  double top_p_value,
                                                                  bool deterministic) {
    check_probs(probs);
    check_uniform(probs, uniform_samples, kUniformDimsFilter);
    TORCH_CHECK(!top_k.defined() ||
                    (top_k.is_cuda() && top_k.is_contiguous() &&
                     top_k.scalar_type() == torch::kInt32 &&
                     top_k.device() == probs.device() && top_k.numel() == probs.size(0)),
                "top_k must be a contiguous CUDA int32 tensor with shape [batch]");
    TORCH_CHECK(!top_p.defined() ||
                    (top_p.is_cuda() && top_p.is_contiguous() &&
                     top_p.scalar_type() == torch::kFloat32 &&
                     top_p.device() == probs.device() && top_p.numel() == probs.size(0)),
                "top_p must be a contiguous CUDA float32 tensor with shape [batch]");

    c10::cuda::CUDAGuard guard(probs.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(probs.device().index()).stream();
    auto samples = make_samples(probs);
    auto success = make_success(probs);
    auto status = flashinfer::sampling::TopKTopPSamplingFromProb<float, int>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<float*>(uniform_samples.data_ptr()),
        top_k.defined() ? static_cast<int*>(top_k.data_ptr()) : nullptr,
        top_p.defined() ? static_cast<float*>(top_p.data_ptr()) : nullptr,
        static_cast<int*>(samples.data_ptr()),
        static_cast<bool*>(success.data_ptr()),
        static_cast<uint32_t>(probs.size(0)),
        static_cast<int>(top_k_value),
        static_cast<float>(top_p_value),
        static_cast<uint32_t>(probs.size(1)),
        static_cast<uint32_t>(uniform_samples.size(0)),
        deterministic,
        stream);
    check_status(status, "FlashInfer TopKTopPSamplingFromProb");
    return {samples, success};
}

}  // namespace sglang
