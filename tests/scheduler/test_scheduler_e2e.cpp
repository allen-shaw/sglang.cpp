#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <tokenizers_cpp.h>

#include "sglang/scheduler/scheduler.h"

namespace sglang {
namespace {

ModelConfig make_test_model_config() {
  ModelConfig config;
  config.num_layers = 1;
  config.num_qo_heads = 1;
  config.num_kv_heads = 1;
  config.head_dim = 128;
  config.hidden_size = 128;
  config.vocab_size = 64;
  config.intermediate_size = 256;
  config.rms_norm_eps = 1e-5F;
  config.rotary_config = RotaryConfig{128, 128, 32, 10000.0F};
  config.hidden_act = "silu";
  config.tie_word_embeddings = false;
  config.num_experts = 0;
  config.num_experts_per_tok = 0;
  config.moe_intermediate_size = 0;
  config.norm_topk_prob = false;
  config.model_type = "llama";
  config.architectures = {"LlamaForCausalLM"};
  config.eos_token_id = -1;
  return config;
}

SchedulerConfig make_test_scheduler_config() {
  SchedulerConfig config;
  config.dtype = torch::kBFloat16;
  config.device = torch::Device(torch::kCUDA, 0);
  config.max_running_req = 4;
  config.page_size = 1;
  config.memory_ratio = 0.01F;
  config.use_dummy_weight = true;
  config.enable_cuda_graph = false;
  config.max_seq_len_override = 32;
  config.num_pages_override = 32;
  config.model_config_override = make_test_model_config();
  config.max_extend_tokens = 16;
  config.cache_type = "radix";
  config.enable_overlap_scheduling = false;
  return config;
}

SchedulerConfig make_test_scheduler_config_with_graph() {
  auto config = make_test_scheduler_config();
  config.enable_cuda_graph = true;
  config.cuda_graph_batch_sizes = {1, 2, 4};
  config.cuda_graph_max_batch_size = 4;
  return config;
}

SchedulerConfig make_overlap_scheduler_config() {
  auto config = make_test_scheduler_config();
  config.enable_overlap_scheduling = true;
  return config;
}

SchedulerConfig make_small_budget_scheduler_config(int max_extend_tokens,
                                                   int max_running_req = 4) {
  auto config = make_test_scheduler_config();
  config.max_extend_tokens = max_extend_tokens;
  config.max_running_req = max_running_req;
  return config;
}

SchedulerConfig make_scaling_scheduler_config(int concurrency) {
  auto config = make_test_scheduler_config();
  config.max_running_req = concurrency;
  config.max_extend_tokens = std::max(128, concurrency * 8);
  config.max_seq_len_override = 16;
  config.num_pages_override = std::max(64, concurrency * 32);
  return config;
}

bool is_cuda_oom(const std::exception& error) {
  const std::string message = error.what();
  return message.find("out of memory") != std::string::npos ||
         message.find("CUDA error: out of memory") != std::string::npos ||
         message.find("CUDA out of memory") != std::string::npos;
}

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

SchedulerConfig make_real_scheduler_config(const std::string& model_path) {
  SchedulerConfig config;
  config.model_path = model_path;
  config.dtype = torch::kBFloat16;
  config.device = torch::Device(torch::kCUDA, 0);
  config.max_running_req = 4;
  config.page_size = 1;
  config.memory_ratio = 0.2F;
  config.use_dummy_weight = false;
  config.enable_cuda_graph = false;
  config.max_extend_tokens = 128;
  config.cache_type = "radix";
  config.enable_overlap_scheduling = false;
  return config;
}

GenerateRequest make_request(uint64_t uid,
                             std::vector<int32_t> input_ids,
                             int max_new_tokens) {
  GenerateRequest request;
  request.uid = uid;
  request.input_ids = torch::tensor(input_ids, torch::kInt32);
  request.sampling_params.temperature = 0.0F;
  request.sampling_params.top_p = 1.0F;
  request.sampling_params.top_k = -1;
  request.sampling_params.max_new_tokens = max_new_tokens;
  return request;
}

std::unordered_map<uint64_t, std::vector<DetokenizeMsg>>
group_by_uid(const std::vector<DetokenizeMsg>& replies) {
  std::unordered_map<uint64_t, std::vector<DetokenizeMsg>> grouped;
  for (const auto& reply : replies) {
    grouped[reply.uid].push_back(reply);
  }
  return grouped;
}

void expect_request_finished(const std::vector<DetokenizeMsg>& replies,
                             uint64_t uid,
                             size_t expected_tokens) {
  auto grouped = group_by_uid(replies);
  auto it = grouped.find(uid);
  ASSERT_NE(it, grouped.end());
  ASSERT_EQ(it->second.size(), expected_tokens);
  for (size_t i = 0; i < it->second.size(); ++i) {
    EXPECT_EQ(it->second[i].uid, uid);
    EXPECT_EQ(it->second[i].finished, i + 1 == it->second.size());
  }
}

std::unique_ptr<tokenizers::Tokenizer> load_tokenizer(const std::string& tokenizer_path) {
  std::ifstream file(tokenizer_path);
  std::stringstream buffer;
  buffer << file.rdbuf();
  return tokenizers::Tokenizer::FromBlobJSON(buffer.str());
}

bool contains_expected_answer(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text.find("paris") != std::string::npos ||
         text.find("巴黎") != std::string::npos;
}

bool contains_expected_city(std::string text, const std::string& city_ascii,
                            const std::string& city_cjk = "") {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (!city_ascii.empty() && text.find(city_ascii) != std::string::npos) {
    return true;
  }
  return !city_cjk.empty() && text.find(city_cjk) != std::string::npos;
}

std::vector<int32_t> collect_generated_ids_for_uid(const std::vector<DetokenizeMsg>& replies,
                                                   uint64_t uid) {
  std::vector<int32_t> generated_ids;
  for (const auto& reply : replies) {
    if (reply.uid == uid) {
      generated_ids.push_back(reply.next_token);
    }
  }
  return generated_ids;
}

TEST(SchedulerE2ETest, SingleRequestRunsToCompletion) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_test_scheduler_config());
  auto request = make_request(/*uid=*/77, {11, 22, 33}, /*max_new_tokens=*/3);

  scheduler.submit(request);
  auto replies = scheduler.run_until_idle();

  ASSERT_EQ(replies.size(), 3);
  EXPECT_FALSE(scheduler.has_work());
  expect_request_finished(replies, request.uid, /*expected_tokens=*/3);
}

TEST(SchedulerE2ETest, SingleRequestRunsToCompletionWithCudaGraphDecode) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_test_scheduler_config_with_graph());
  auto request = make_request(/*uid=*/78, {11, 22, 33}, /*max_new_tokens=*/3);

  scheduler.submit(request);
  auto replies = scheduler.run_until_idle();

  ASSERT_EQ(replies.size(), 3);
  EXPECT_FALSE(scheduler.has_work());
  expect_request_finished(replies, request.uid, /*expected_tokens=*/3);
}

TEST(SchedulerE2ETest, TwoRequestsRunToCompletionWithCudaGraphDecode) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_test_scheduler_config_with_graph());
  auto request_a = make_request(/*uid=*/181, {11, 12, 13}, /*max_new_tokens=*/2);
  auto request_b = make_request(/*uid=*/182, {21, 22, 23, 24}, /*max_new_tokens=*/3);

  scheduler.submit(request_a);
  scheduler.submit(request_b);
  auto replies = scheduler.run_until_idle();

  ASSERT_EQ(replies.size(), 5);
  EXPECT_FALSE(scheduler.has_work());
  expect_request_finished(replies, request_a.uid, /*expected_tokens=*/2);
  expect_request_finished(replies, request_b.uid, /*expected_tokens=*/3);
}

TEST(SchedulerE2ETest, ConcurrencySweepWithCudaGraphDecode) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  for (int concurrency : {8, 16}) {
    auto config = make_scaling_scheduler_config(concurrency);
    config.enable_cuda_graph = true;
    config.cuda_graph_batch_sizes = {1, 2, 4, 8, 16};
    config.cuda_graph_max_batch_size = 16;

    Scheduler scheduler(config);
    for (int i = 0; i < concurrency; ++i) {
      scheduler.submit(make_request(
          /*uid=*/5000 + i, {1, 2, 3, static_cast<int32_t>(i + 4)}, /*max_new_tokens=*/2));
    }

    auto replies = scheduler.run_until_idle();
    ASSERT_EQ(replies.size(), static_cast<size_t>(concurrency * 2));
    EXPECT_FALSE(scheduler.has_work());
    for (int i = 0; i < concurrency; ++i) {
      expect_request_finished(replies, /*uid=*/5000 + i, /*expected_tokens=*/2);
    }
  }
}

TEST(SchedulerE2ETest, StaggeredLongRequestsWithCudaGraphDecode) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  auto config = make_test_scheduler_config_with_graph();
  config.max_running_req = 16;
  config.max_extend_tokens = 4096;
  config.max_seq_len_override = 4096;
  config.num_pages_override = 4096;
  config.memory_ratio = 0.05F;
  config.cuda_graph_batch_sizes = {1, 2, 4, 8, 16};
  config.cuda_graph_max_batch_size = 16;

  Scheduler scheduler(config);
  std::vector<DetokenizeMsg> replies;
  for (int i = 0; i < 8; ++i) {
    std::vector<int32_t> prompt;
    const int prompt_len = 64 + i * 32;
    prompt.reserve(prompt_len);
    for (int j = 0; j < prompt_len; ++j) {
      prompt.push_back(static_cast<int32_t>((j + i) % 60));
    }
    scheduler.submit(make_request(/*uid=*/7000 + i, std::move(prompt), /*max_new_tokens=*/16));
    if (i % 2 == 1) {
      auto partial = scheduler.step();
      EXPECT_FALSE(partial.empty());
      replies.insert(replies.end(), partial.begin(), partial.end());
    }
  }

  auto tail_replies = scheduler.run_until_idle();
  replies.insert(replies.end(), tail_replies.begin(), tail_replies.end());
  EXPECT_FALSE(scheduler.has_work());
  auto grouped = group_by_uid(replies);
  ASSERT_EQ(grouped.size(), 8u);
  for (int i = 0; i < 8; ++i) {
    expect_request_finished(replies, /*uid=*/7000 + i, /*expected_tokens=*/16);
  }
}

TEST(SchedulerE2ETest, TwoRequestsSameTickRunToCompletion) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_test_scheduler_config());
  auto request_a = make_request(/*uid=*/101, {11, 12, 13}, /*max_new_tokens=*/2);
  auto request_b = make_request(/*uid=*/202, {21, 22, 23, 24}, /*max_new_tokens=*/3);

  scheduler.submit(request_a);
  scheduler.submit(request_b);
  auto replies = scheduler.run_until_idle();

  ASSERT_EQ(replies.size(), 5);
  EXPECT_FALSE(scheduler.has_work());
  expect_request_finished(replies, request_a.uid, /*expected_tokens=*/2);
  expect_request_finished(replies, request_b.uid, /*expected_tokens=*/3);
}

TEST(SchedulerE2ETest, ThreeRequestsSameTickRunToCompletion) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_test_scheduler_config());
  auto request_a = make_request(/*uid=*/211, {1, 2, 3}, /*max_new_tokens=*/2);
  auto request_b = make_request(/*uid=*/212, {4, 5, 6, 7}, /*max_new_tokens=*/3);
  auto request_c = make_request(/*uid=*/213, {8, 9, 10}, /*max_new_tokens=*/1);

  scheduler.submit(request_a);
  scheduler.submit(request_b);
  scheduler.submit(request_c);
  auto replies = scheduler.run_until_idle();

  ASSERT_EQ(replies.size(), 6);
  EXPECT_FALSE(scheduler.has_work());
  expect_request_finished(replies, request_a.uid, /*expected_tokens=*/2);
  expect_request_finished(replies, request_b.uid, /*expected_tokens=*/3);
  expect_request_finished(replies, request_c.uid, /*expected_tokens=*/1);
}

TEST(SchedulerE2ETest, StaggeredArrivalDuringDecodeStillCompletesBothRequests) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_test_scheduler_config());
  auto request_a = make_request(/*uid=*/301, {7, 8, 9}, /*max_new_tokens=*/4);
  auto request_b = make_request(/*uid=*/302, {17, 18, 19}, /*max_new_tokens=*/2);

  scheduler.submit(request_a);
  auto first_step = scheduler.step();

  ASSERT_EQ(first_step.size(), 1);
  EXPECT_EQ(first_step[0].uid, request_a.uid);
  EXPECT_FALSE(first_step[0].finished);
  EXPECT_TRUE(scheduler.has_work());

  scheduler.submit(request_b);
  auto remaining_replies = scheduler.run_until_idle();

  std::vector<DetokenizeMsg> all_replies = first_step;
  all_replies.insert(all_replies.end(), remaining_replies.begin(), remaining_replies.end());

  ASSERT_EQ(all_replies.size(), 6);
  EXPECT_FALSE(scheduler.has_work());
  expect_request_finished(all_replies, request_a.uid, /*expected_tokens=*/4);
  expect_request_finished(all_replies, request_b.uid, /*expected_tokens=*/2);
}

TEST(SchedulerE2ETest, DifferentMaxNewTokensFinishIndependently) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_test_scheduler_config());
  auto short_request = make_request(/*uid=*/401, {1, 2, 3}, /*max_new_tokens=*/1);
  auto long_request = make_request(/*uid=*/402, {4, 5, 6}, /*max_new_tokens=*/4);

  scheduler.submit(short_request);
  scheduler.submit(long_request);
  auto replies = scheduler.run_until_idle();

  ASSERT_EQ(replies.size(), 5);
  EXPECT_FALSE(scheduler.has_work());
  expect_request_finished(replies, short_request.uid, /*expected_tokens=*/1);
  expect_request_finished(replies, long_request.uid, /*expected_tokens=*/4);
}

TEST(SchedulerE2ETest, OverlapSingleRequestTailFlushesToCompletion) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_overlap_scheduler_config());
  auto request = make_request(/*uid=*/451, {1, 2, 3}, /*max_new_tokens=*/1);

  scheduler.submit(request);
  auto first_step = scheduler.step();

  EXPECT_TRUE(first_step.empty());
  EXPECT_TRUE(scheduler.has_work());

  auto replies = scheduler.run_until_idle();
  ASSERT_EQ(replies.size(), 1);
  EXPECT_FALSE(scheduler.has_work());
  expect_request_finished(replies, request.uid, /*expected_tokens=*/1);
}

TEST(SchedulerE2ETest, OverlapAndNonOverlapReturnSameGreedyTokenCounts) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  auto non_overlap_config = make_test_scheduler_config();
  auto overlap_config = non_overlap_config;
  overlap_config.enable_overlap_scheduling = true;

  std::vector<GenerateRequest> requests = {
      make_request(/*uid=*/461, {1, 2, 3}, /*max_new_tokens=*/2),
      make_request(/*uid=*/462, {4, 5, 6, 7}, /*max_new_tokens=*/3),
      make_request(/*uid=*/463, {8, 9, 10}, /*max_new_tokens=*/1),
  };

  Scheduler non_overlap_scheduler(non_overlap_config);
  for (const auto& request : requests) {
    non_overlap_scheduler.submit(request);
  }
  auto non_overlap_replies = non_overlap_scheduler.run_until_idle();

  Scheduler overlap_scheduler(overlap_config);
  for (const auto& request : requests) {
    overlap_scheduler.submit(request);
  }
  auto overlap_replies = overlap_scheduler.run_until_idle();

  ASSERT_EQ(overlap_replies.size(), non_overlap_replies.size());
  EXPECT_FALSE(overlap_scheduler.has_work());
  for (const auto& request : requests) {
    EXPECT_EQ(collect_generated_ids_for_uid(overlap_replies, request.uid).size(),
              collect_generated_ids_for_uid(non_overlap_replies, request.uid).size());
    expect_request_finished(overlap_replies,
                            request.uid,
                            collect_generated_ids_for_uid(non_overlap_replies, request.uid).size());
  }
}

TEST(SchedulerE2ETest, AbortRunningDecodeRequestStopsFurtherTokensForThatRequest) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_test_scheduler_config());
  auto request_a = make_request(/*uid=*/501, {30, 31, 32}, /*max_new_tokens=*/4);
  auto request_b = make_request(/*uid=*/502, {40, 41, 42}, /*max_new_tokens=*/2);

  scheduler.submit(request_a);
  auto first_step = scheduler.step();

  ASSERT_EQ(first_step.size(), 1);
  EXPECT_EQ(first_step[0].uid, request_a.uid);
  EXPECT_FALSE(first_step[0].finished);

  scheduler.abort(request_a.uid);
  scheduler.submit(request_b);
  auto remaining_replies = scheduler.run_until_idle();

  std::vector<DetokenizeMsg> all_replies = first_step;
  all_replies.insert(all_replies.end(), remaining_replies.begin(), remaining_replies.end());
  auto grouped = group_by_uid(all_replies);

  ASSERT_EQ(grouped[request_a.uid].size(), 1);
  EXPECT_FALSE(grouped[request_a.uid][0].finished);
  ASSERT_EQ(grouped[request_b.uid].size(), 2);
  EXPECT_FALSE(scheduler.has_work());
  expect_request_finished(all_replies, request_b.uid, /*expected_tokens=*/2);
}

TEST(SchedulerE2ETest, OverlapAbortInFlightSuppressesExtraTokenAndRecyclesSlot) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  auto config = make_overlap_scheduler_config();
  config.max_running_req = 1;
  Scheduler scheduler(config);
  auto request_a = make_request(/*uid=*/551, {30, 31, 32}, /*max_new_tokens=*/4);
  auto request_b = make_request(/*uid=*/552, {40, 41, 42}, /*max_new_tokens=*/2);

  scheduler.submit(request_a);
  EXPECT_TRUE(scheduler.step().empty());
  auto first_replies = scheduler.step();

  ASSERT_EQ(first_replies.size(), 1);
  EXPECT_EQ(first_replies[0].uid, request_a.uid);
  EXPECT_FALSE(first_replies[0].finished);

  scheduler.abort(request_a.uid);
  scheduler.submit(request_b);
  auto remaining_replies = scheduler.run_until_idle();

  std::vector<DetokenizeMsg> all_replies = first_replies;
  all_replies.insert(all_replies.end(), remaining_replies.begin(), remaining_replies.end());
  auto grouped = group_by_uid(all_replies);

  ASSERT_EQ(grouped[request_a.uid].size(), 1);
  ASSERT_EQ(grouped[request_b.uid].size(), 2);
  EXPECT_FALSE(scheduler.has_work());
  expect_request_finished(all_replies, request_b.uid, /*expected_tokens=*/2);
}

TEST(SchedulerE2ETest, AbortPendingRequestRemovesItBeforeScheduling) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_test_scheduler_config());
  auto request_a = make_request(/*uid=*/601, {50, 51, 52}, /*max_new_tokens=*/2);
  auto request_b = make_request(/*uid=*/602, {60, 61, 62}, /*max_new_tokens=*/3);

  scheduler.submit(request_a);
  scheduler.submit(request_b);
  scheduler.abort(request_b.uid);
  auto replies = scheduler.run_until_idle();

  ASSERT_EQ(replies.size(), 2);
  EXPECT_FALSE(scheduler.has_work());
  expect_request_finished(replies, request_a.uid, /*expected_tokens=*/2);
  EXPECT_EQ(group_by_uid(replies).count(request_b.uid), 0U);
}

TEST(SchedulerE2ETest, ChunkedPrefillUnderBudgetPressureStillCompletesRequest) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_small_budget_scheduler_config(/*max_extend_tokens=*/2));
  auto request = make_request(/*uid=*/651, {1, 2, 3, 4, 5, 6}, /*max_new_tokens=*/2);

  scheduler.submit(request);
  auto first_step = scheduler.step();

  EXPECT_TRUE(first_step.empty());
  EXPECT_TRUE(scheduler.has_work());

  auto replies = scheduler.run_until_idle();
  ASSERT_EQ(replies.size(), 2);
  EXPECT_FALSE(scheduler.has_work());
  expect_request_finished(replies, request.uid, /*expected_tokens=*/2);
}

TEST(SchedulerE2ETest, SharedPrefixRequestUsesCacheAndAvoidsInitialChunking) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_small_budget_scheduler_config(/*max_extend_tokens=*/1));
  auto warmup_request = make_request(/*uid=*/661, {9, 8, 7, 6, 5}, /*max_new_tokens=*/1);
  auto cached_request = make_request(/*uid=*/662, {9, 8, 7, 6, 4}, /*max_new_tokens=*/1);

  scheduler.submit(warmup_request);
  auto warmup_replies = scheduler.run_until_idle();

  ASSERT_EQ(warmup_replies.size(), 1);
  expect_request_finished(warmup_replies, warmup_request.uid, /*expected_tokens=*/1);
  EXPECT_FALSE(scheduler.has_work());

  scheduler.submit(cached_request);
  auto first_step = scheduler.step();

  ASSERT_EQ(first_step.size(), 1);
  EXPECT_EQ(first_step[0].uid, cached_request.uid);
  EXPECT_TRUE(first_step[0].finished);
  EXPECT_FALSE(scheduler.has_work());
}

TEST(SchedulerE2ETest, SequentialBatchesRecycleRunningSlots) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(
      make_small_budget_scheduler_config(/*max_extend_tokens=*/16, /*max_running_req=*/2));
  auto first_batch_a = make_request(/*uid=*/671, {1, 2, 3}, /*max_new_tokens=*/2);
  auto first_batch_b = make_request(/*uid=*/672, {4, 5, 6}, /*max_new_tokens=*/2);
  auto second_batch_a = make_request(/*uid=*/673, {7, 8, 9}, /*max_new_tokens=*/2);
  auto second_batch_b = make_request(/*uid=*/674, {10, 11, 12}, /*max_new_tokens=*/2);

  scheduler.submit(first_batch_a);
  scheduler.submit(first_batch_b);
  auto first_batch_replies = scheduler.run_until_idle();

  ASSERT_EQ(first_batch_replies.size(), 4);
  expect_request_finished(first_batch_replies, first_batch_a.uid, /*expected_tokens=*/2);
  expect_request_finished(first_batch_replies, first_batch_b.uid, /*expected_tokens=*/2);
  EXPECT_FALSE(scheduler.has_work());

  scheduler.submit(second_batch_a);
  scheduler.submit(second_batch_b);
  auto second_batch_replies = scheduler.run_until_idle();

  ASSERT_EQ(second_batch_replies.size(), 4);
  expect_request_finished(second_batch_replies, second_batch_a.uid, /*expected_tokens=*/2);
  expect_request_finished(second_batch_replies, second_batch_b.uid, /*expected_tokens=*/2);
  EXPECT_FALSE(scheduler.has_work());
}

TEST(SchedulerE2ETest, ChunkedPrefillCanArriveWhileAnotherRequestIsDecoding) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_small_budget_scheduler_config(/*max_extend_tokens=*/3));
  auto decode_request = make_request(/*uid=*/681, {1, 2, 3}, /*max_new_tokens=*/4);
  auto long_prefill_request = make_request(/*uid=*/682, {10, 11, 12, 13, 14, 15}, /*max_new_tokens=*/2);

  scheduler.submit(decode_request);
  auto first_step = scheduler.step();

  ASSERT_EQ(first_step.size(), 1);
  EXPECT_EQ(first_step[0].uid, decode_request.uid);
  EXPECT_FALSE(first_step[0].finished);

  scheduler.submit(long_prefill_request);
  auto second_step = scheduler.step();

  EXPECT_TRUE(second_step.empty());
  EXPECT_TRUE(scheduler.has_work());

  auto remaining_replies = scheduler.run_until_idle();
  std::vector<DetokenizeMsg> all_replies = first_step;
  all_replies.insert(all_replies.end(), remaining_replies.begin(), remaining_replies.end());

  ASSERT_EQ(all_replies.size(), 6);
  expect_request_finished(all_replies, decode_request.uid, /*expected_tokens=*/4);
  expect_request_finished(all_replies, long_prefill_request.uid, /*expected_tokens=*/2);
  EXPECT_FALSE(scheduler.has_work());
}

TEST(SchedulerE2ETest, RealModelPromptPrintsFranceCapitalAnswer) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for real-model Scheduler smoke test";
  }

  const std::string model_path = find_qwen3_model_path();
  if (model_path.empty()) {
    GTEST_SKIP()
        << "Qwen3 model not found. Set QWEN3_MODEL_PATH or download Qwen/Qwen3-0.6B";
  }

  const std::string tokenizer_path = model_path + "/tokenizer.json";
  if (!std::filesystem::exists(tokenizer_path)) {
    GTEST_SKIP() << "tokenizer.json not found at: " << tokenizer_path;
  }

  auto tokenizer = load_tokenizer(tokenizer_path);
  Scheduler scheduler(make_real_scheduler_config(model_path));

  const std::string prompt =
      "Question: What is the capital of France?\n"
      "Answer in one short sentence.";
  auto prompt_ids = tokenizer->Encode(prompt);
  ASSERT_FALSE(prompt_ids.empty());

  GenerateRequest request;
  request.uid = 701;
  request.input_ids = torch::tensor(prompt_ids, torch::kInt32).contiguous();
  request.sampling_params.temperature = 0.0F;
  request.sampling_params.top_p = 1.0F;
  request.sampling_params.top_k = -1;
  request.sampling_params.max_new_tokens = 24;

  scheduler.submit(request);

  std::vector<int32_t> generated_ids;
  std::cout << "\nPrompt: " << prompt << std::endl;
  while (scheduler.has_work()) {
    auto replies = scheduler.step();
    for (const auto& reply : replies) {
      generated_ids.push_back(reply.next_token);
      std::cout << reply.next_token << " ";
    }
  }
  std::cout << std::endl;

  std::string generated_text = tokenizer->Decode(generated_ids);
  std::cout << "Decoded output: " << generated_text << std::endl;

  EXPECT_FALSE(generated_text.empty());
  EXPECT_TRUE(contains_expected_answer(generated_text))
      << "Expected answer to mention Paris/巴黎, got: " << generated_text;
}

TEST(SchedulerE2ETest, RealModelTwoRequestsAnswerDifferentCapitalQuestions) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for real-model Scheduler smoke test";
  }

  const std::string model_path = find_qwen3_model_path();
  if (model_path.empty()) {
    GTEST_SKIP()
        << "Qwen3 model not found. Set QWEN3_MODEL_PATH or download Qwen/Qwen3-0.6B";
  }

  const std::string tokenizer_path = model_path + "/tokenizer.json";
  if (!std::filesystem::exists(tokenizer_path)) {
    GTEST_SKIP() << "tokenizer.json not found at: " << tokenizer_path;
  }

  auto tokenizer = load_tokenizer(tokenizer_path);
  Scheduler scheduler(make_real_scheduler_config(model_path));

  const std::string france_prompt =
      "Question: What is the capital of France?\n"
      "Answer in one short sentence.";
  const std::string germany_prompt =
      "Question: What is the capital of Germany?\n"
      "Answer in one short sentence.";

  auto france_ids = tokenizer->Encode(france_prompt);
  auto germany_ids = tokenizer->Encode(germany_prompt);
  ASSERT_FALSE(france_ids.empty());
  ASSERT_FALSE(germany_ids.empty());

  auto france_request = make_request(/*uid=*/711, france_ids, /*max_new_tokens=*/24);
  auto germany_request = make_request(/*uid=*/712, germany_ids, /*max_new_tokens=*/24);

  scheduler.submit(france_request);
  scheduler.submit(germany_request);
  auto replies = scheduler.run_until_idle();

  auto france_generated = collect_generated_ids_for_uid(replies, france_request.uid);
  auto germany_generated = collect_generated_ids_for_uid(replies, germany_request.uid);
  auto france_text = tokenizer->Decode(france_generated);
  auto germany_text = tokenizer->Decode(germany_generated);

  std::cout << "\nFrance prompt output: " << france_text << std::endl;
  std::cout << "Germany prompt output: " << germany_text << std::endl;

  expect_request_finished(replies, france_request.uid, /*expected_tokens=*/24);
  expect_request_finished(replies, germany_request.uid, /*expected_tokens=*/24);
  EXPECT_TRUE(contains_expected_city(france_text, "paris", "巴黎"))
      << "Expected France answer to mention Paris/巴黎, got: " << france_text;
  EXPECT_TRUE(contains_expected_city(germany_text, "berlin", "柏林"))
      << "Expected Germany answer to mention Berlin/柏林, got: " << germany_text;
}

TEST(SchedulerE2ETest, ConcurrencySweepUpToOom) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  const std::vector<int> concurrency_levels = {8, 16, 64, 128};
  int max_completed_concurrency = 0;

  for (int concurrency : concurrency_levels) {
    SCOPED_TRACE("scheduler_concurrency=" + std::to_string(concurrency));
    try {
      Scheduler scheduler(make_scaling_scheduler_config(concurrency));
      for (int i = 0; i < concurrency; ++i) {
        scheduler.submit(
            make_request(/*uid=*/10000 + static_cast<uint64_t>(i),
                         {1, 2, 3, static_cast<int32_t>(10 + (i % 13))},
                         /*max_new_tokens=*/2));
      }

      auto replies = scheduler.run_until_idle();
      ASSERT_EQ(replies.size(), static_cast<size_t>(concurrency * 2));
      EXPECT_FALSE(scheduler.has_work());
      for (int i = 0; i < concurrency; ++i) {
        expect_request_finished(replies,
                                /*uid=*/10000 + static_cast<uint64_t>(i),
                                /*expected_tokens=*/2);
      }
      max_completed_concurrency = concurrency;
      std::cout << "[scheduler sweep] completed concurrency=" << concurrency << std::endl;
    } catch (const c10::Error& error) {
      if (is_cuda_oom(error)) {
        std::cout << "[scheduler sweep] hit OOM at concurrency=" << concurrency << std::endl;
        break;
      }
      throw;
    } catch (const std::runtime_error& error) {
      if (is_cuda_oom(error)) {
        std::cout << "[scheduler sweep] hit OOM at concurrency=" << concurrency << std::endl;
        break;
      }
      throw;
    }
  }

  EXPECT_GE(max_completed_concurrency, 8);
}

}  // namespace
}  // namespace sglang
