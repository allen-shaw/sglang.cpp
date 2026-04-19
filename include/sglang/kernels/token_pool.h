#pragma once

#include <torch/torch.h>

namespace sglang {

void write_token_pool(const torch::Tensor& token_pool,
                      const torch::Tensor& req_indices,
                      const torch::Tensor& positions,
                      const torch::Tensor& tokens);

}  // namespace sglang
