#pragma once

#include <memory>
#include <unordered_map>
#include <string>

#include <tokenizers_cpp.h>

#include "sglang/tokenizer/tokenizer_base.h"

namespace sglang {

class HFTokenizeManager : public TokenizeManagerBase {
 public:
  HFTokenizeManager(const std::string& tokenizer_json_path);
  ~HFTokenizeManager() override = default;

  std::vector<torch::Tensor> tokenize(const std::vector<TokenizeMsg>& msgs) override;

 private:
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

struct DecodeStatus {
  std::vector<int32_t> decoded_ids;
  std::string decoded_str;
  size_t read_offset = 0;
  size_t surr_offset = 0;
  size_t sent_offset = 0;
};

class HFDetokenizeManager : public DetokenizeManagerBase {
 public:
  HFDetokenizeManager(const std::string& tokenizer_json_path);
  ~HFDetokenizeManager() override = default;

  std::vector<std::string> detokenize(const std::vector<DetokenizeMsg>& msgs) override;

 private:
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unordered_map<uint64_t, DecodeStatus> decode_map_;
};

}  // namespace sglang
