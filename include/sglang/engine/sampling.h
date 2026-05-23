#pragma once

#include <torch/torch.h>

namespace sglang {

struct FlashInferSamplingResult {
    torch::Tensor samples;
    torch::Tensor success;
};

torch::Tensor flashinfer_sample_from_probs(const torch::Tensor& probs,
                                           const torch::Tensor& uniform_samples,
                                           bool deterministic);

FlashInferSamplingResult flashinfer_top_k_sample_from_probs(const torch::Tensor& probs,
                                                            const torch::Tensor& uniform_samples,
                                                            const torch::Tensor& top_k,
                                                            int64_t top_k_value,
                                                            bool deterministic);

FlashInferSamplingResult flashinfer_top_p_sample_from_probs(const torch::Tensor& probs,
                                                            const torch::Tensor& uniform_samples,
                                                            const torch::Tensor& top_p,
                                                            double top_p_value,
                                                            bool deterministic);

FlashInferSamplingResult flashinfer_top_k_top_p_sample_from_probs(const torch::Tensor& probs,
                                                                  const torch::Tensor& uniform_samples,
                                                                  const torch::Tensor& top_k,
                                                                  double top_k_value,
                                                                  const torch::Tensor& top_p,
                                                                  double top_p_value,
                                                                  bool deterministic);

}  // namespace sglang
