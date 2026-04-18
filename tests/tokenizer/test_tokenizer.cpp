#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "sglang/message/tokenizer_msg.h"
#include "sglang/tokenizer/hf_tokenizer.h"

using namespace sglang;

namespace {

std::string find_qwen3_tokenizer_path() {
  if (const char* env_p = std::getenv("QWEN3_MODEL_PATH")) {
    const auto path = std::string(env_p) + "/tokenizer.json";
    if (std::filesystem::exists(path)) {
      return path;
    }
  }

  const char* home = std::getenv("HOME");
  if (!home) {
    return "";
  }

  const std::string hf_dir =
      std::string(home) + "/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots";
  if (!std::filesystem::exists(hf_dir)) {
    return "";
  }

  for (const auto& entry : std::filesystem::directory_iterator(hf_dir)) {
    if (entry.is_directory()) {
      const auto tokenizer_path = entry.path().string() + "/tokenizer.json";
      if (std::filesystem::exists(tokenizer_path)) {
        return tokenizer_path;
      }
    }
  }
  return "";
}

}  // namespace

class TokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* env_p = std::getenv("TEST_TOKENIZER_JSON");
    if (env_p) {
      tokenizer_path_ = env_p;
    } else {
      tokenizer_path_ = "tokenizer.json";
    }

    std::ifstream f(tokenizer_path_);
    if (!f.good()) {
      GTEST_SKIP() << "Skipping tokenizer tests: " << tokenizer_path_ << " not found. Set TEST_TOKENIZER_JSON env var to a valid tokenizer.json";
    }
  }

  std::string tokenizer_path_;
};

TEST_F(TokenizerTest, BasicTokenize) {
  HFTokenizeManager tokenize_manager(tokenizer_path_);
  
  std::vector<TokenizeMsg> msgs;
  TokenizeMsg msg1;
  msg1.uid = 1;
  msg1.text = "Hello, world!";
  msgs.push_back(msg1);

  auto tensors = tokenize_manager.tokenize(msgs);
  ASSERT_EQ(tensors.size(), 1);
  EXPECT_GT(tensors[0].numel(), 0);
  EXPECT_EQ(tensors[0].scalar_type(), torch::kInt32);
}

TEST_F(TokenizerTest, BasicDetokenize) {
  HFTokenizeManager tokenize_manager(tokenizer_path_);
  HFDetokenizeManager detokenize_manager(tokenizer_path_);

  // First tokenize to get some valid ids
  std::vector<TokenizeMsg> t_msgs;
  TokenizeMsg t_msg;
  t_msg.uid = 1;
  t_msg.text = "Hello, world! Welcome to sglang.";
  t_msgs.push_back(t_msg);

  auto tensors = tokenize_manager.tokenize(t_msgs);
  ASSERT_EQ(tensors.size(), 1);
  int32_t* ids_ptr = tensors[0].data_ptr<int32_t>();
  int num_ids = tensors[0].numel();

  // Feed them one by one
  std::string full_response = "";
  for (int i = 0; i < num_ids; ++i) {
    std::vector<DetokenizeMsg> d_msgs;
    DetokenizeMsg d_msg;
    d_msg.uid = 1;
    d_msg.next_token = ids_ptr[i];
    d_msg.finished = (i == num_ids - 1);
    d_msgs.push_back(d_msg);

    auto inc_strs = detokenize_manager.detokenize(d_msgs);
    ASSERT_EQ(inc_strs.size(), 1);
    full_response += inc_strs[0];
  }

  // Expect roughly the same text (ignoring special tokens formatting differences if any)
  EXPECT_TRUE(full_response.find("Hello") != std::string::npos);
}

TEST(TokenizerChatTemplateTest, StructuredMessagesUseChatTemplatePath) {
  const auto tokenizer_path = find_qwen3_tokenizer_path();
  if (tokenizer_path.empty()) {
    GTEST_SKIP() << "Skipping chat-template tokenizer test: Qwen3 tokenizer.json not found";
  }

  HFTokenizeManager tokenize_manager(tokenizer_path);
  std::vector<TokenizeMsg> msgs;
  TokenizeMsg msg;
  msg.uid = 1;
  msg.text = std::vector<ChatMessage>{
      ChatMessage{"system", "You are a concise assistant."},
      ChatMessage{"user", "What is the capital of France?"},
  };
  msgs.push_back(std::move(msg));

  auto tensors = tokenize_manager.tokenize(msgs);
  ASSERT_EQ(tensors.size(), 1);
  ASSERT_GT(tensors[0].numel(), 0);

  auto ids_cpu = tensors[0].to(torch::kCPU).contiguous();
  std::vector<int32_t> ids(ids_cpu.data_ptr<int32_t>(),
                           ids_cpu.data_ptr<int32_t>() + ids_cpu.numel());

  HFDetokenizeManager detokenize_manager(tokenizer_path);
  std::vector<std::string> pieces;
  for (size_t i = 0; i < ids.size(); ++i) {
    DetokenizeMsg d_msg;
    d_msg.uid = 77;
    d_msg.next_token = ids[i];
    d_msg.finished = (i + 1 == ids.size());
    auto out = detokenize_manager.detokenize({d_msg});
    ASSERT_EQ(out.size(), 1);
    pieces.push_back(out.front());
  }

  std::string rendered;
  for (const auto& piece : pieces) {
    rendered += piece;
  }
  EXPECT_NE(rendered.find("France"), std::string::npos);
}
