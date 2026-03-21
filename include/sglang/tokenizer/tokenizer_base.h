#pragma once

#include <vector>
#include <string>

#include <torch/torch.h>

#include "sglang/message/tokenizer_msg.h"

namespace sglang {

/// Abstract base class for tokenization.
class TokenizeManagerBase {
 public:
  virtual ~TokenizeManagerBase() = default;

  /// Tokenizes a batch of messages into a list of 1D tensors.
  virtual std::vector<torch::Tensor> tokenize(const std::vector<TokenizeMsg>& msgs) = 0;
};

/// Abstract base class for detokenization.
class DetokenizeManagerBase {
 public:
  virtual ~DetokenizeManagerBase() = default;

  /// Detokenizes a batch of upcoming tokens and returns the incremental strings.
  virtual std::vector<std::string> detokenize(const std::vector<DetokenizeMsg>& msgs) = 0;
};

}  // namespace sglang
