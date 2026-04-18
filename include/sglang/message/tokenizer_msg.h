#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "sglang/core/sampling_params.h"

namespace sglang {

struct ChatMessage {
  std::string role;
  std::string content;
};

using TokenizeInput = std::variant<std::string, std::vector<ChatMessage>>;

/// Message to request tokenization of text.
struct TokenizeMsg {
  uint64_t uid = 0;
  TokenizeInput text;
  SamplingParams sampling_params;
};

/// Message to request detokenization of the next token.
struct DetokenizeMsg {
  uint64_t uid = 0;
  int32_t next_token = 0;
  bool finished = false;
};

}  // namespace sglang
