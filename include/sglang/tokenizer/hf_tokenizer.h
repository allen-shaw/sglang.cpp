#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <tokenizers_cpp.h>

#include "sglang/message/tokenizer_msg.h"
#include "sglang/tokenizer/tokenizer_base.h"

namespace sglang {

class ChatTemplateRenderer {
 public:
  explicit ChatTemplateRenderer(const std::string& tokenizer_json_path);

  std::string render(const std::vector<ChatMessage>& messages,
                     bool add_generation_prompt = true) const;
  bool available() const { return mode_.has_value(); }

 private:
  enum class TemplateMode {
    kQwenIm,
  };

  std::optional<TemplateMode> mode_;
};

class HFTokenizeManager : public TokenizeManagerBase {
 public:
  HFTokenizeManager(const std::string& tokenizer_json_path);
  ~HFTokenizeManager() override = default;

  torch::Tensor tokenize_one(const std::string& prompt) const;
  torch::Tensor tokenize_one(const std::vector<ChatMessage>& messages) const;
  std::vector<torch::Tensor> tokenize(const std::vector<TokenizeMsg>& msgs) override;

 private:
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<ChatTemplateRenderer> chat_template_renderer_;
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
