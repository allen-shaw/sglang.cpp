#include "sglang/tokenizer/hf_tokenizer.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <codecvt>
#include <locale>

namespace sglang {

namespace {

std::string read_file(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open tokenizer json file: " + path);
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
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
    return text; // Invalid UTF-8, just return
  }

  if (u32_str.empty()) return text;

  // If the last token is a CJK character, we print the characters.
  if (is_chinese_char(u32_str.back())) {
    return text;
  }
  
  // Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
  if (u32_str.size() > 1 && is_chinese_char(u32_str[u32_str.size() - 2])) {
    return conv.to_bytes(u32_str.substr(0, u32_str.size() - 1));
  }
  
  // Otherwise, prints until the last space char
  size_t last_space = text.rfind(' ');
  if (last_space != std::string::npos) {
    return text.substr(0, last_space + 1);
  }
  return text;
}

} // namespace

HFTokenizeManager::HFTokenizeManager(const std::string& tokenizer_json_path) {
  std::string blob = read_file(tokenizer_json_path);
  tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);
}

std::vector<torch::Tensor> HFTokenizeManager::tokenize(const std::vector<TokenizeMsg>& msgs) {
  std::vector<torch::Tensor> results;
  results.reserve(msgs.size());

  for (const auto& msg : msgs) {
    // Note: Python logic supports chat templates using apply_chat_template.
    // tokenizers-cpp currently might not strictly reproduce all hf apply_chat_template
    // natively or cleanly. For now we assume msg.text is plain string.
    std::string prompt = msg.text;
    
    std::vector<int32_t> ids = tokenizer_->Encode(prompt);
    
    auto tensor = torch::empty({static_cast<int64_t>(ids.size())}, torch::kInt32);
    std::memcpy(tensor.data_ptr<int32_t>(), ids.data(), ids.size() * sizeof(int32_t));
    
    results.push_back(std::move(tensor));
  }
  
  return results;
}

HFDetokenizeManager::HFDetokenizeManager(const std::string& tokenizer_json_path) {
  std::string blob = read_file(tokenizer_json_path);
  tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);
}

std::vector<std::string> HFDetokenizeManager::detokenize(const std::vector<DetokenizeMsg>& msgs) {
  std::vector<std::vector<int32_t>> read_ids;
  std::vector<std::vector<int32_t>> surr_ids;
  
  // Prepare batch decode requests
  for (const auto& msg : msgs) {
    if (decode_map_.find(msg.uid) == decode_map_.end()) {
      decode_map_[msg.uid] = DecodeStatus{};
    }
    auto& s = decode_map_[msg.uid];
    
    // We assume eos condition is parsed/checked, but here we just append non-finished tokens
    if (!msg.finished) {
      s.decoded_ids.push_back(msg.next_token);
    }
    
    std::vector<int32_t> read_slice(s.decoded_ids.begin() + s.surr_offset, s.decoded_ids.end());
    std::vector<int32_t> surr_slice(s.decoded_ids.begin() + s.surr_offset, s.decoded_ids.begin() + s.read_offset);
    
    read_ids.push_back(std::move(read_slice));
    surr_ids.push_back(std::move(surr_slice));
  }

  // Batch decode
  // Note: tokenizers-cpp might not have a direct BatchDecode, but doing it sequentially is fine for now
  std::vector<std::string> read_texts;
  std::vector<std::string> surr_texts;
  for (const auto& ids : read_ids) read_texts.push_back(tokenizer_->Decode(ids));
  for (const auto& ids : surr_ids) surr_texts.push_back(tokenizer_->Decode(ids));

  std::vector<std::string> incremental_strs;
  for (size_t i = 0; i < msgs.size(); ++i) {
    const auto& msg = msgs[i];
    auto& s = decode_map_[msg.uid];
    
    const std::string& read_str = read_texts[i];
    const std::string& surr_str = surr_texts[i];
    
    std::string new_text = read_str.substr(surr_str.length());
    
    // Streaming chunk: update the decode status
    // 0xEF 0xBF 0xBD is the replacement character "" in utf-8
    bool ends_with_replacement = (new_text.size() >= 3 && 
                                  new_text.substr(new_text.size() - 3) == "\xEF\xBF\xBD");
    
    if (!new_text.empty() && !ends_with_replacement) {
      std::string output_str = s.decoded_str + new_text;
      s.decoded_str = output_str;
      s.surr_offset = s.read_offset;
      s.read_offset = s.decoded_ids.size();
    } else {
      new_text = find_printable_text(new_text);
      std::string output_str = s.decoded_str + new_text;
      // We don't advance offsets here since it's incomplete
    }

    std::string output_str_full = s.decoded_str; 
    if (ends_with_replacement) {
      output_str_full += new_text; // Temporary
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
