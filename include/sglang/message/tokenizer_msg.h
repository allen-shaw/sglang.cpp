#pragma once

#include <cstdint>
#include <string>

#include "sglang/core/sampling_params.h"

namespace sglang {

/// Message to request tokenization of text.
struct TokenizeMsg {
  uint64_t uid = 0;
  std::string text;
  // TODO: Add support for chat templates (e.g., list of dicts) if needed in the future
  SamplingParams sampling_params;
};

/// Message to request detokenization of the next token.
struct DetokenizeMsg {
  uint64_t uid = 0;
  int32_t next_token = 0;
  bool finished = false;
};

}  // namespace sglang
