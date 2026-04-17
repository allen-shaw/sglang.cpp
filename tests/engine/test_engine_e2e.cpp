#include <gtest/gtest.h>

#include "sglang/engine/engine.h"

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

EngineConfig make_test_engine_config() {
  EngineConfig config;
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
  return config;
}

torch::Tensor make_positions_for_req(const Req& req, const torch::Device& device) {
  return torch::arange(
      req.cached_len,
      req.device_len(),
      torch::TensorOptions().dtype(torch::kInt32).device(device));
}

torch::Tensor make_out_loc_for_req(const Req& req, const torch::Device& device) {
  return torch::arange(
      req.cached_len,
      req.device_len(),
      torch::TensorOptions().dtype(torch::kInt32).device(device));
}

torch::Tensor make_input_ids_for_req(const Req& req, const torch::Device& device) {
  return req.input_ids.slice(0, req.cached_len, req.device_len()).to(device, torch::kInt32);
}

TEST(EngineE2ETest, SingleRequestRunsToCompletion) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Engine E2E test";
  }

  Engine engine(make_test_engine_config());

  auto req = std::make_shared<Req>();
  req->req_id = 123;
  req->table_idx = 0;
  req->input_ids = torch::tensor({3, 5, 7}, torch::kInt32);
  req->cached_len = 0;
  req->output_len = 3;
  req->initialize_runtime_state();

  std::vector<int32_t> generated_tokens;
  bool first_step = true;
  while (true) {
    Batch batch;
    batch.reqs = {req};
    batch.phase = first_step ? BatchPhase::Prefill : BatchPhase::Decode;
    engine.pad_batch(batch);

    batch.positions = make_positions_for_req(*req, engine.device());
    batch.out_loc = make_out_loc_for_req(*req, engine.device());
    batch.input_ids = make_input_ids_for_req(*req, engine.device());
    engine.prepare_attention_metadata(batch);

    auto sampling_args = engine.prepare_sampling_args(batch);
    auto output = engine.forward_batch(batch, sampling_args);
    output.synchronize();

    auto next_token = output.next_tokens_cpu[0].to(torch::kInt32).reshape({1});
    generated_tokens.push_back(next_token.item<int32_t>());
    req->append_host(next_token);

    if (!req->can_decode()) {
      break;
    }
    first_step = false;
  }

  EXPECT_EQ(generated_tokens.size(), 3);
  EXPECT_EQ(req->input_ids.size(0), 6);
  EXPECT_EQ(req->cached_len, 5);
  EXPECT_EQ(req->device_len(), 6);
  EXPECT_EQ(req->max_device_len(), 6);
  EXPECT_FALSE(req->can_decode());
}

}  // namespace
}  // namespace sglang
