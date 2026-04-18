#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "sglang/llm/llm.h"

namespace sglang {
namespace {

std::string find_qwen3_model_path() {
  if (const char* env_path = std::getenv("QWEN3_MODEL_PATH")) {
    return std::string(env_path);
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
      return entry.path().string();
    }
  }
  return "";
}

bool contains_paris(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text.find("paris") != std::string::npos || text.find("巴黎") != std::string::npos;
}

TEST(LLMTest, DummyWeightSinglePromptReturnsTokens) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for LLM tests";
  }

  ServerArgs args;
  args.model_path = find_qwen3_model_path();
  if (args.model_path.empty()) {
    GTEST_SKIP() << "Qwen3 tokenizer/model path not found";
  }
  args.dtype = torch::kBFloat16;
  args.num_tokenizer_threads = 1;
  args.max_running_req = 2;
  args.memory_ratio = 0.01F;
  args.page_size = 1;
  args.num_pages_override = 32;
  args.max_seq_len_override = 32;
  args.max_extend_tokens = 8;
  args.use_dummy_weight = true;

  LLM llm(args);
  SamplingParams params;
  params.temperature = 0.0F;
  params.top_p = 1.0F;
  params.top_k = -1;
  params.max_new_tokens = 4;

  auto result = llm.generate("Hello from sglang.cpp", params);
  EXPECT_FALSE(result.token_ids.empty());
  EXPECT_TRUE(result.finished);
}

TEST(LLMTest, RealModelFrancePromptMentionsParis) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for LLM tests";
  }

  auto model_path = find_qwen3_model_path();
  if (model_path.empty()) {
    GTEST_SKIP() << "Qwen3 model path not found";
  }

  ServerArgs args;
  args.model_path = model_path;
  args.dtype = torch::kBFloat16;
  args.num_tokenizer_threads = 1;
  args.max_running_req = 2;
  args.memory_ratio = 0.2F;
  args.page_size = 1;
  args.max_extend_tokens = 64;

  LLM llm(args);
  SamplingParams params;
  params.temperature = 0.0F;
  params.top_p = 1.0F;
  params.top_k = -1;
  params.max_new_tokens = 24;

  auto result = llm.generate(
      "Question: What is the capital of France?\nAnswer in one short sentence.", params);
  EXPECT_TRUE(contains_paris(result.text)) << "Expected Paris/巴黎 in: " << result.text;
}

}  // namespace
}  // namespace sglang
