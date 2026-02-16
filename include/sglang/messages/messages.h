#pragma once

#include "proto/sglang.pb.h"

namespace sglang {

// Re-export specific message types if needed
using GenerateReq = sglang::GenerateReq;
using GenerateResp = sglang::GenerateResp;
using SamplingParams = sglang::SamplingParams;

// Helper function to check if sampling params indicate greedy decoding
bool is_greedy(const SamplingParams& params);

} // namespace sglang
