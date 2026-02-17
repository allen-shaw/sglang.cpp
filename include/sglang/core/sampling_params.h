#pragma once

#include "proto/sglang.pb.h"

namespace sglang {

// Re-export specific message types if needed
// Redundant aliases removed as they conflict with protobuf generated classes in the same namespace

// Helper function to check if sampling params indicate greedy decoding
bool is_greedy(const SamplingParams& params);

} // namespace sglang
