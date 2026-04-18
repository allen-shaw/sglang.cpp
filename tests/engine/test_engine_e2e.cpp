#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "sglang/engine/engine.h"
#include "sglang/scheduler/prefill.h"

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

EngineConfig make_test_engine_config_with_graph() {
  auto config = make_test_engine_config();
  config.enable_cuda_graph = true;
  config.cuda_graph_batch_sizes = {1, 2, 4};
  config.cuda_graph_max_batch_size = 4;
  return config;
}

EngineConfig make_scaling_engine_config(int concurrency) {
  auto config = make_test_engine_config();
  config.max_running_req = concurrency;
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

std::shared_ptr<Req> make_req(uint64_t req_id,
                              int table_idx,
                              std::vector<int32_t> input_ids,
                              int max_new_tokens) {
  auto req = std::make_shared<Req>();
  req->req_id = req_id;
  req->table_idx = table_idx;
  req->input_ids = torch::tensor(input_ids, torch::kInt32);
  req->cached_len = 0;
  req->output_len = max_new_tokens;
  req->initialize_runtime_state();
  return req;
}

torch::Tensor make_positions(const Batch& batch, const torch::Device& device) {
  std::vector<int32_t> positions_host;
  for (const auto& req : batch.padded_reqs) {
    for (int pos = req->cached_len; pos < req->device_len(); ++pos) {
      positions_host.push_back(pos);
    }
  }
  return torch::tensor(
      positions_host,
      torch::TensorOptions().dtype(torch::kInt32).device(device));
}

std::pair<torch::Tensor, torch::Tensor> make_input_tuple(const Batch& batch,
                                                         const torch::Device& device) {
  std::vector<int64_t> req_mapping;
  std::vector<int64_t> positions;
  for (const auto& req : batch.padded_reqs) {
    for (int pos = req->cached_len; pos < req->device_len(); ++pos) {
      req_mapping.push_back(req->table_idx);
      positions.push_back(pos);
    }
  }
  return {
      torch::tensor(req_mapping, torch::TensorOptions().dtype(torch::kInt64).device(device)),
      torch::tensor(positions, torch::TensorOptions().dtype(torch::kInt64).device(device)),
  };
}

std::pair<torch::Tensor, torch::Tensor> make_write_tuple(const Batch& batch,
                                                         const torch::Device& device) {
  std::vector<int64_t> req_mapping;
  std::vector<int64_t> write_positions;
  for (const auto& req : batch.reqs) {
    req_mapping.push_back(req->table_idx);
    write_positions.push_back(req->can_decode() ? req->device_len() : -1);
  }
  return {
      torch::tensor(req_mapping, torch::TensorOptions().dtype(torch::kInt64).device(device)),
      torch::tensor(write_positions, torch::TensorOptions().dtype(torch::kInt64).device(device)),
  };
}

struct PreparedBatch {
  torch::Tensor req_mapping;
  torch::Tensor token_positions;
  torch::Tensor write_req_mapping;
  torch::Tensor write_positions;
};

PreparedBatch prepare_batch(Engine& engine,
                            TableManager& table_manager,
                            CacheManager& cache_manager,
                            Batch& batch) {
  engine.pad_batch(batch);
  cache_manager.allocate_paged(batch.reqs);
  batch.positions = make_positions(batch, engine.device());
  auto input_tuple = make_input_tuple(batch, engine.device());
  auto write_tuple = make_write_tuple(batch, engine.device());
  batch.out_loc = engine.page_table().index({input_tuple.first, input_tuple.second});
  batch.input_ids = table_manager.token_pool().index({input_tuple.first, input_tuple.second});
  engine.prepare_attention_metadata(batch);
  return {input_tuple.first, input_tuple.second, write_tuple.first, write_tuple.second};
}

void append_sampled_tokens(const ForwardOutput& output, const std::vector<std::shared_ptr<Req>>& reqs) {
  for (size_t i = 0; i < reqs.size(); ++i) {
    auto next_token = output.next_tokens_cpu.index({static_cast<int64_t>(i)}).to(torch::kInt32).reshape({1});
    reqs[i]->append_host(next_token);
  }
}

void write_next_tokens_to_pool(const ForwardOutput& output,
                               const PreparedBatch& prepared,
                               TableManager& table_manager) {
  auto valid_mask = prepared.write_positions.ge(0);
  if (!valid_mask.any().item<bool>()) {
    return;
  }
  auto valid_req = prepared.write_req_mapping.index({valid_mask});
  auto valid_pos = prepared.write_positions.index({valid_mask});
  auto valid_tokens = output.next_tokens_gpu.index({valid_mask});
  table_manager.token_pool().index_put_({valid_req, valid_pos}, valid_tokens);
}

void copy_prompt_to_pool(TableManager& table_manager, const Req& req) {
  auto device_ids = table_manager.token_pool()[req.table_idx].slice(0, 0, req.device_len());
  device_ids.copy_(req.input_ids.to(device_ids.device()));
}

TEST(EngineE2ETest, SingleRequestRunsToCompletion) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Engine E2E test";
  }

  Engine engine(make_test_engine_config());
  TableManager table_manager(/*max_running_reqs=*/4, engine.page_table());
  CacheManager cache_manager(engine.num_pages(), /*page_size=*/1, engine.page_table(), "radix");

  auto req = make_req(/*req_id=*/123, table_manager.allocate(), {3, 5, 7}, /*max_new_tokens=*/3);
  copy_prompt_to_pool(table_manager, *req);

  std::vector<int32_t> generated_tokens;
  bool first_step = true;
  while (true) {
    Batch batch;
    batch.reqs = {req};
    batch.phase = first_step ? BatchPhase::Prefill : BatchPhase::Decode;
    auto prepared = prepare_batch(engine, table_manager, cache_manager, batch);

    auto sampling_args = engine.prepare_sampling_args(batch);
    auto output = engine.forward_batch(batch, sampling_args);
    output.synchronize();
    write_next_tokens_to_pool(output, prepared, table_manager);

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

TEST(EngineE2ETest, SingleRequestRunsToCompletionWithCudaGraphDecode) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Engine E2E test";
  }

  Engine engine(make_test_engine_config_with_graph());
  TableManager table_manager(/*max_running_reqs=*/4, engine.page_table());
  CacheManager cache_manager(engine.num_pages(), /*page_size=*/1, engine.page_table(), "radix");

  auto req = make_req(/*req_id=*/124, table_manager.allocate(), {3, 5, 7}, /*max_new_tokens=*/3);
  copy_prompt_to_pool(table_manager, *req);

  bool first_step = true;
  while (true) {
    Batch batch;
    batch.reqs = {req};
    batch.phase = first_step ? BatchPhase::Prefill : BatchPhase::Decode;
    auto prepared = prepare_batch(engine, table_manager, cache_manager, batch);

    auto sampling_args = engine.prepare_sampling_args(batch);
    auto output = engine.forward_batch(batch, sampling_args);
    output.synchronize();
    write_next_tokens_to_pool(output, prepared, table_manager);

    auto next_token = output.next_tokens_cpu[0].to(torch::kInt32).reshape({1});
    req->append_host(next_token);
    if (!req->can_decode()) {
      break;
    }
    first_step = false;
  }

  EXPECT_EQ(req->input_ids.size(0), 6);
  EXPECT_EQ(req->device_len(), 6);
  EXPECT_FALSE(req->can_decode());
}

TEST(EngineE2ETest, TwoRequestsSameBatchFinishIndependently) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Engine E2E test";
  }

  Engine engine(make_test_engine_config());
  TableManager table_manager(/*max_running_reqs=*/4, engine.page_table());
  CacheManager cache_manager(engine.num_pages(), /*page_size=*/1, engine.page_table(), "radix");

  auto req_a = make_req(/*req_id=*/201, table_manager.allocate(), {3, 5, 7}, /*max_new_tokens=*/2);
  auto req_b =
      make_req(/*req_id=*/202, table_manager.allocate(), {11, 13, 17, 19}, /*max_new_tokens=*/3);
  copy_prompt_to_pool(table_manager, *req_a);
  copy_prompt_to_pool(table_manager, *req_b);

  bool first_step = true;
  while (req_a->can_decode() || req_b->can_decode()) {
    Batch batch;
    batch.phase = first_step ? BatchPhase::Prefill : BatchPhase::Decode;
    if (req_a->can_decode()) {
      batch.reqs.push_back(req_a);
    }
    if (req_b->can_decode()) {
      batch.reqs.push_back(req_b);
    }
    auto prepared = prepare_batch(engine, table_manager, cache_manager, batch);

    auto sampling_args = engine.prepare_sampling_args(batch);
    auto output = engine.forward_batch(batch, sampling_args);
    output.synchronize();
    write_next_tokens_to_pool(output, prepared, table_manager);
    append_sampled_tokens(output, batch.reqs);
    first_step = false;
  }

  EXPECT_EQ(req_a->input_ids.size(0), 5);
  EXPECT_EQ(req_b->input_ids.size(0), 7);
  EXPECT_EQ(req_a->max_device_len(), 5);
  EXPECT_EQ(req_b->max_device_len(), 7);
  EXPECT_FALSE(req_a->can_decode());
  EXPECT_FALSE(req_b->can_decode());
}

TEST(EngineE2ETest, StaggeredSecondRequestJoinsBeforeSharedDecode) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Engine E2E test";
  }

  Engine engine(make_test_engine_config());
  TableManager table_manager(/*max_running_reqs=*/4, engine.page_table());
  CacheManager cache_manager(engine.num_pages(), /*page_size=*/1, engine.page_table(), "radix");

  auto req_a = make_req(/*req_id=*/301, table_manager.allocate(), {2, 4, 6}, /*max_new_tokens=*/4);
  auto req_b = make_req(/*req_id=*/302, table_manager.allocate(), {8, 10, 12}, /*max_new_tokens=*/2);
  copy_prompt_to_pool(table_manager, *req_a);
  copy_prompt_to_pool(table_manager, *req_b);

  {
    Batch batch;
    batch.reqs = {req_a};
    batch.phase = BatchPhase::Prefill;
    auto prepared = prepare_batch(engine, table_manager, cache_manager, batch);

    auto sampling_args = engine.prepare_sampling_args(batch);
    auto output = engine.forward_batch(batch, sampling_args);
    output.synchronize();
    write_next_tokens_to_pool(output, prepared, table_manager);
    append_sampled_tokens(output, batch.reqs);
  }

  ASSERT_TRUE(req_a->can_decode());
  ASSERT_TRUE(req_b->can_decode());

  {
    Batch batch;
    batch.reqs = {req_b};
    batch.phase = BatchPhase::Prefill;
    auto prepared = prepare_batch(engine, table_manager, cache_manager, batch);

    auto sampling_args = engine.prepare_sampling_args(batch);
    auto output = engine.forward_batch(batch, sampling_args);
    output.synchronize();
    write_next_tokens_to_pool(output, prepared, table_manager);
    append_sampled_tokens(output, batch.reqs);
  }

  while (req_a->can_decode() || req_b->can_decode()) {
    Batch batch;
    batch.phase = BatchPhase::Decode;
    if (req_a->can_decode()) {
      batch.reqs.push_back(req_a);
    }
    if (req_b->can_decode()) {
      batch.reqs.push_back(req_b);
    }
    auto prepared = prepare_batch(engine, table_manager, cache_manager, batch);

    auto sampling_args = engine.prepare_sampling_args(batch);
    auto output = engine.forward_batch(batch, sampling_args);
    output.synchronize();
    write_next_tokens_to_pool(output, prepared, table_manager);
    append_sampled_tokens(output, batch.reqs);
  }

  EXPECT_EQ(req_a->input_ids.size(0), 7);
  EXPECT_EQ(req_b->input_ids.size(0), 5);
  EXPECT_FALSE(req_a->can_decode());
  EXPECT_FALSE(req_b->can_decode());
}

TEST(EngineE2ETest, ConcurrencySweepUpToOom) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for Engine E2E test";
  }

  const std::vector<int> concurrency_levels = {8, 16, 64, 128};
  int max_completed_concurrency = 0;

  for (int concurrency : concurrency_levels) {
    SCOPED_TRACE("engine_concurrency=" + std::to_string(concurrency));
    try {
      Engine engine(make_scaling_engine_config(concurrency));
      TableManager table_manager(concurrency, engine.page_table());
      CacheManager cache_manager(engine.num_pages(), /*page_size=*/1, engine.page_table(), "radix");

      std::vector<std::shared_ptr<Req>> reqs;
      reqs.reserve(concurrency);
      for (int i = 0; i < concurrency; ++i) {
        auto req = make_req(
            /*req_id=*/1000 + static_cast<uint64_t>(i),
            table_manager.allocate(),
            {1, 2, 3, static_cast<int32_t>(4 + (i % 8))},
            /*max_new_tokens=*/2);
        copy_prompt_to_pool(table_manager, *req);
        reqs.push_back(std::move(req));
      }

      bool first_step = true;
      while (std::any_of(reqs.begin(), reqs.end(), [](const auto& req) { return req->can_decode(); })) {
        Batch batch;
        batch.phase = first_step ? BatchPhase::Prefill : BatchPhase::Decode;
        for (const auto& req : reqs) {
          if (req->can_decode()) {
            batch.reqs.push_back(req);
          }
        }
        auto prepared = prepare_batch(engine, table_manager, cache_manager, batch);
        auto sampling_args = engine.prepare_sampling_args(batch);
        auto output = engine.forward_batch(batch, sampling_args);
        output.synchronize();
        write_next_tokens_to_pool(output, prepared, table_manager);
        append_sampled_tokens(output, batch.reqs);
        first_step = false;
      }

      for (const auto& req : reqs) {
        EXPECT_FALSE(req->can_decode());
        EXPECT_EQ(req->input_ids.size(0), 6);
      }
      max_completed_concurrency = concurrency;
      std::cout << "[engine sweep] completed concurrency=" << concurrency << std::endl;
    } catch (const c10::Error& error) {
      if (is_cuda_oom(error)) {
        std::cout << "[engine sweep] hit OOM at concurrency=" << concurrency << std::endl;
        break;
      }
      throw;
    } catch (const std::runtime_error& error) {
      if (is_cuda_oom(error)) {
        std::cout << "[engine sweep] hit OOM at concurrency=" << concurrency << std::endl;
        break;
      }
      throw;
    }
  }

  EXPECT_GE(max_completed_concurrency, 8);
}

}  // namespace
}  // namespace sglang
