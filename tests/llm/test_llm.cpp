#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <string>
#include <thread>
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

bool contains_berlin(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text.find("berlin") != std::string::npos || text.find("柏林") != std::string::npos;
}

bool is_cuda_oom(const std::exception& error) {
  const std::string message = error.what();
  return message.find("out of memory") != std::string::npos ||
         message.find("CUDA error: out of memory") != std::string::npos ||
         message.find("CUDA out of memory") != std::string::npos;
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

TEST(LLMTest, ConcurrentGenerateReturnsIndependentResults) {
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
  args.num_tokenizer_threads = 2;
  args.max_running_req = 4;
  args.memory_ratio = 0.2F;
  args.page_size = 1;
  args.max_extend_tokens = 64;

  LLM llm(args);
  SamplingParams params;
  params.temperature = 0.0F;
  params.top_p = 1.0F;
  params.top_k = -1;
  params.max_new_tokens = 24;

  auto france_future = std::async(std::launch::async, [&]() {
    return llm.generate(
        "Question: What is the capital of France?\nAnswer in one short sentence.", params);
  });
  auto germany_future = std::async(std::launch::async, [&]() {
    return llm.generate(
        "Question: What is the capital of Germany?\nAnswer in one short sentence.", params);
  });

  auto france = france_future.get();
  auto germany = germany_future.get();

  EXPECT_TRUE(contains_paris(france.text)) << "Expected Paris/巴黎 in: " << france.text;
  EXPECT_TRUE(contains_berlin(germany.text)) << "Expected Berlin/柏林 in: " << germany.text;
  EXPECT_TRUE(france.finished);
  EXPECT_TRUE(germany.finished);
  EXPECT_NE(france.uid, germany.uid);
}

TEST(LLMTest, BatchConcurrencySweepUpToOom) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for LLM tests";
  }

  auto model_path = find_qwen3_model_path();
  if (model_path.empty()) {
    GTEST_SKIP() << "Qwen3 model path not found";
  }

  const std::vector<int> concurrency_levels = {8, 16, 64, 128};
  int max_completed_concurrency = 0;

  for (int concurrency : concurrency_levels) {
    SCOPED_TRACE("llm_concurrency=" + std::to_string(concurrency));
    try {
      ServerArgs args;
      args.model_path = model_path;
      args.dtype = torch::kBFloat16;
      args.num_tokenizer_threads = std::min(concurrency, 8);
      args.max_running_req = concurrency;
      args.memory_ratio = 0.01F;
      args.page_size = 1;
      args.max_extend_tokens = std::max(128, concurrency * 8);
      args.num_pages_override = std::max(64, concurrency * 32);
      args.max_seq_len_override = 64;
      args.use_dummy_weight = true;

      LLM llm(args);
      SamplingParams params;
      params.temperature = 0.0F;
      params.top_p = 1.0F;
      params.top_k = -1;
      params.max_new_tokens = 2;
      params.ignore_eos = true;

      std::vector<std::string> prompts;
      std::vector<SamplingParams> params_list;
      prompts.reserve(concurrency);
      params_list.reserve(concurrency);
      for (int i = 0; i < concurrency; ++i) {
        prompts.push_back("Hello #" + std::to_string(i));
        params_list.push_back(params);
      }

      auto results = llm.generate(prompts, params_list);
      ASSERT_EQ(results.size(), static_cast<size_t>(concurrency));
      for (const auto& result : results) {
        EXPECT_TRUE(result.finished);
        EXPECT_GE(result.token_ids.size(), 1);
        EXPECT_LE(result.token_ids.size(), 2);
      }

      max_completed_concurrency = concurrency;
      std::cout << "[llm sweep] completed concurrency=" << concurrency << std::endl;
    } catch (const c10::Error& error) {
      if (is_cuda_oom(error)) {
        std::cout << "[llm sweep] hit OOM at concurrency=" << concurrency << std::endl;
        break;
      }
      throw;
    } catch (const std::runtime_error& error) {
      if (is_cuda_oom(error)) {
        std::cout << "[llm sweep] hit OOM at concurrency=" << concurrency << std::endl;
        break;
      }
      throw;
    }
  }

  EXPECT_GE(max_completed_concurrency, 8);
}

}  // namespace
}  // namespace sglang
