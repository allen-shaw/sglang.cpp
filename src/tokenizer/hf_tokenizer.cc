#include "sglang/tokenizer/hf_tokenizer.h"

#include <codecvt>
#include <cstring>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace sglang {

namespace {

std::string read_file(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

std::string dirname(const std::string& path) {
  auto pos = path.find_last_of("/\\");
  return pos == std::string::npos ? "." : path.substr(0, pos);
}

torch::Tensor make_tensor_from_ids(const std::vector<int32_t>& ids) {
  auto tensor = torch::empty({static_cast<int64_t>(ids.size())}, torch::kInt32);
  if (!ids.empty()) {
    std::memcpy(tensor.data_ptr<int32_t>(), ids.data(), ids.size() * sizeof(int32_t));
  }
  return tensor;
}

bool is_chinese_char(int cp) {
  if ((cp >= 0x4E00 && cp <= 0x9FFF) ||
      (cp >= 0x3400 && cp <= 0x4DBF) ||
      (cp >= 0x20000 && cp <= 0x2A6DF) ||
      (cp >= 0x2A700 && cp <= 0x2B73F) ||
      (cp >= 0x2B740 && cp <= 0x2B81F) ||
      (cp >= 0x2B820 && cp <= 0x2CEAF) ||
      (cp >= 0xF900 && cp <= 0xFAFF) ||
      (cp >= 0x2F800 && cp <= 0x2FA1F)) {
    return true;
  }
  return false;
}

std::string find_printable_text(const std::string& text) {
  if (text.empty()) return text;
  if (text.back() == '\n') return text;

  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::u32string u32_str;
  try {
    u32_str = conv.from_bytes(text);
  } catch (...) {
    return text;
  }

  if (u32_str.empty()) return text;
  if (is_chinese_char(u32_str.back())) {
    return text;
  }
  if (u32_str.size() > 1 && is_chinese_char(u32_str[u32_str.size() - 2])) {
    return conv.to_bytes(u32_str.substr(0, u32_str.size() - 1));
  }

  size_t last_space = text.rfind(' ');
  if (last_space != std::string::npos) {
    return text.substr(0, last_space + 1);
  }
  return text;
}

std::string render_qwen_im_template(const std::vector<ChatMessage>& messages,
                                    bool add_generation_prompt) {
  std::ostringstream out;
  for (const auto& message : messages) {
    out << "<|im_start|>" << message.role << "\n"
        << message.content << "<|im_end|>\n";
  }
  if (add_generation_prompt) {
    out << "<|im_start|>assistant\n";
  }
  return out.str();
}

}  // namespace

ChatTemplateRenderer::ChatTemplateRenderer(const std::string& tokenizer_json_path) {
  try {
    const auto config_path = dirname(tokenizer_json_path) + "/tokenizer_config.json";
    auto config = nlohmann::json::parse(read_file(config_path));
    if (config.contains("chat_template") && config["chat_template"].is_string()) {
      const auto template_str = config["chat_template"].get<std::string>();
      if (template_str.find("<|im_start|>") != std::string::npos &&
          template_str.find("<|im_end|>") != std::string::npos) {
        mode_ = TemplateMode::kQwenIm;
      }
    }
  } catch (...) {
    mode_.reset();
  }
}

std::string ChatTemplateRenderer::render(const std::vector<ChatMessage>& messages,
                                         bool add_generation_prompt) const {
  if (!mode_.has_value()) {
    std::ostringstream out;
    for (const auto& message : messages) {
      out << message.role << ": " << message.content << "\n";
    }
    if (add_generation_prompt) {
      out << "assistant: ";
    }
    return out.str();
  }

  switch (*mode_) {
    case TemplateMode::kQwenIm:
      return render_qwen_im_template(messages, add_generation_prompt);
  }
  throw std::runtime_error("Unsupported chat template mode");
}

HFTokenizeManager::HFTokenizeManager(const std::string& tokenizer_json_path) {
  std::string blob = read_file(tokenizer_json_path);
  tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);
  chat_template_renderer_ =
      std::make_unique<ChatTemplateRenderer>(tokenizer_json_path);
}

torch::Tensor HFTokenizeManager::tokenize_one(const std::string& prompt) const {
  return make_tensor_from_ids(tokenizer_->Encode(prompt));
}

torch::Tensor HFTokenizeManager::tokenize_one(
    const std::vector<ChatMessage>& messages) const {
  const auto prompt = chat_template_renderer_->render(messages, /*add_generation_prompt=*/true);
  return tokenize_one(prompt);
}

std::vector<torch::Tensor> HFTokenizeManager::tokenize(const std::vector<TokenizeMsg>& msgs) {
  std::vector<torch::Tensor> results;
  results.reserve(msgs.size());

  for (const auto& msg : msgs) {
    if (std::holds_alternative<std::string>(msg.text)) {
      results.push_back(tokenize_one(std::get<std::string>(msg.text)));
    } else {
      results.push_back(tokenize_one(std::get<std::vector<ChatMessage>>(msg.text)));
    }
  }

  return results;
}

HFDetokenizeManager::HFDetokenizeManager(const std::string& tokenizer_json_path) {
  std::string blob = read_file(tokenizer_json_path);
  tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);
}

std::vector<std::string> HFDetokenizeManager::detokenize(
    const std::vector<DetokenizeMsg>& msgs) {
  std::vector<std::vector<int32_t>> read_ids;
  std::vector<std::vector<int32_t>> surr_ids;

  for (const auto& msg : msgs) {
    if (decode_map_.find(msg.uid) == decode_map_.end()) {
      decode_map_[msg.uid] = DecodeStatus{};
    }
    auto& s = decode_map_[msg.uid];

    if (msg.next_token >= 0) {
      s.decoded_ids.push_back(msg.next_token);
    }

    std::vector<int32_t> read_slice(s.decoded_ids.begin() + s.surr_offset, s.decoded_ids.end());
    std::vector<int32_t> surr_slice(
        s.decoded_ids.begin() + s.surr_offset, s.decoded_ids.begin() + s.read_offset);

    read_ids.push_back(std::move(read_slice));
    surr_ids.push_back(std::move(surr_slice));
  }

  std::vector<std::string> read_texts;
  std::vector<std::string> surr_texts;
  for (const auto& ids : read_ids) {
    read_texts.push_back(ids.empty() ? std::string() : tokenizer_->Decode(ids));
  }
  for (const auto& ids : surr_ids) {
    surr_texts.push_back(ids.empty() ? std::string() : tokenizer_->Decode(ids));
  }

  std::vector<std::string> incremental_strs;
  for (size_t i = 0; i < msgs.size(); ++i) {
    const auto& msg = msgs[i];
    auto& s = decode_map_[msg.uid];

    const std::string& read_str = read_texts[i];
    const std::string& surr_str = surr_texts[i];

    std::string new_text = read_str.substr(surr_str.length());
    bool ends_with_replacement =
        (new_text.size() >= 3 &&
         new_text.substr(new_text.size() - 3) == "\xEF\xBF\xBD");

    if (!new_text.empty() && !ends_with_replacement) {
      s.decoded_str += new_text;
      s.surr_offset = s.read_offset;
      s.read_offset = s.decoded_ids.size();
    } else {
      new_text = find_printable_text(new_text);
    }

    std::string output_str_full = s.decoded_str;
    if (ends_with_replacement) {
      output_str_full += new_text;
      output_str_full = find_printable_text(output_str_full);
    }

    std::string incremental_output = output_str_full.substr(s.sent_offset);
    s.sent_offset = output_str_full.length();
    incremental_strs.push_back(incremental_output);

    if (msg.finished) {
      decode_map_.erase(msg.uid);
    }
  }

  return incremental_strs;
}

}  // namespace sglang
