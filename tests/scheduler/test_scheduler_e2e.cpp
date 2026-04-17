#include <gtest/gtest.h>

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

TEST(SchedulerE2ETest, SingleRequestRunsToCompletion) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Scheduler E2E test";
  }

  Scheduler scheduler(make_test_scheduler_config());

  GenerateRequest request;
  request.uid = 77;
  request.input_ids = torch::tensor({11, 22, 33}, torch::kInt32);
  request.sampling_params.temperature = 0.0F;
  request.sampling_params.top_p = 1.0F;
  request.sampling_params.top_k = -1;
  request.sampling_params.max_new_tokens = 3;

  scheduler.submit(request);
  auto replies = scheduler.run_until_idle();

  ASSERT_EQ(replies.size(), 3);
  EXPECT_FALSE(scheduler.has_work());
  for (size_t i = 0; i < replies.size(); ++i) {
    EXPECT_EQ(replies[i].uid, request.uid);
    EXPECT_EQ(replies[i].finished, i + 1 == replies.size());
  }
}

}  // namespace
}  // namespace sglang
