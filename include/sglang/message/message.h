#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "sglang/core/sampling_params.h"

namespace sglang {

/// Request message from frontend to backend.
/// Corresponds to Python's UserMsg.
struct GenerateRequest {
  uint64_t uid = 0;
  torch::Tensor input_ids;  // CPU 1D int32 tensor
  SamplingParams sampling_params;
};

/// Response message from backend to frontend.
/// Corresponds to Python's UserReply.
struct GenerateResponse {
  uint64_t uid = 0;
  std::string incremental_output;
  bool finished = false;
};

/// Batch of responses.
struct BatchResponse {
  std::vector<GenerateResponse> data;
};

/// Exit signal message.
struct ExitMessage {};

/// Abort message.
struct AbortMessage {
  uint64_t uid = 0;
};

}  // namespace sglang
