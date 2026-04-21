#include <gtest/gtest.h>

#include "sglang/engine/engine.h"

namespace sglang {
namespace {

std::shared_ptr<Req> make_req(float temperature = 0.0F,
                              int top_k = -1,
                              float top_p = 1.0F) {
  auto req = std::make_shared<Req>();
  req->input_ids = torch::tensor({1}, torch::kInt32);
  req->sampling_params.temperature = temperature;
  req->sampling_params.top_k = top_k;
  req->sampling_params.top_p = top_p;
  req->sampling_params.max_new_tokens = 4;
  req->initialize_runtime_state();
  return req;
}

TEST(SamplerTest, GreedySamplingUsesArgmax) {
  Batch batch;
  batch.reqs = {make_req(), make_req()};

  Sampler sampler(torch::kCPU, 4);
  auto args = sampler.prepare(batch);
  auto logits = torch::tensor({{1.0F, 3.0F, 2.0F, 0.5F},
                               {0.0F, 2.0F, 5.0F, 1.0F}});

  auto next_tokens = sampler.sample(logits, args);

  EXPECT_TRUE(torch::equal(next_tokens, torch::tensor({1, 2}, torch::kLong)));
}

TEST(SamplerTest, TopPSamplingKeepsOnlyAllowedPrefix) {
  torch::manual_seed(0);

  Batch batch;
  batch.reqs = {make_req(0.8F, -1, 0.6F)};

  Sampler sampler(torch::kCPU, 4);
  auto args = sampler.prepare(batch);
  auto logits = torch::tensor({{10.0F, 1.0F, 0.5F, -1.0F}});

  auto next_tokens = sampler.sample(logits, args);

  EXPECT_EQ(next_tokens[0].item<int64_t>(), 0);
}

TEST(SamplerTest, PrepareMaterializesPerRequestSamplingArgs) {
  Batch batch;
  batch.reqs = {
      make_req(0.7F, 4, 0.9F),
      make_req(1.0F, -1, 1.0F),
  };

  Sampler sampler(torch::kCPU, 16);
  auto args = sampler.prepare(batch);

  EXPECT_TRUE(args.temperatures.defined());
  EXPECT_TRUE(args.top_k.defined());
  EXPECT_TRUE(args.top_p.defined());
  EXPECT_EQ(args.temperatures.size(0), 2);
  EXPECT_EQ(args.top_k.size(0), 2);
  EXPECT_EQ(args.top_p.size(0), 2);
}

TEST(SamplerTest, FlashInferSamplingReturnsCudaTokens) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }

  Batch batch;
  batch.reqs = {make_req(0.8F), make_req(1.0F)};

  Sampler sampler(torch::kCUDA, 4);
  auto args = sampler.prepare(batch);
  auto logits = torch::tensor({{1.0F, 3.0F, 2.0F, 0.5F},
                               {0.0F, 2.0F, 5.0F, 1.0F}},
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  auto next_tokens = sampler.sample(logits, args);

  EXPECT_TRUE(next_tokens.is_cuda());
  EXPECT_EQ(next_tokens.scalar_type(), torch::kInt32);
  EXPECT_EQ(next_tokens.size(0), 2);
  auto next_tokens_cpu = next_tokens.cpu();
  EXPECT_GE(next_tokens_cpu.min().item<int32_t>(), 0);
  EXPECT_LT(next_tokens_cpu.max().item<int32_t>(), 4);
}

TEST(SamplerTest, FlashInferTopKOneSamplesArgmax) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }

  Batch batch;
  batch.reqs = {make_req(0.7F, 1, 1.0F), make_req(1.0F, 1, 1.0F)};

  Sampler sampler(torch::kCUDA, 4);
  auto args = sampler.prepare(batch);
  auto logits = torch::tensor({{1.0F, 3.0F, 2.0F, 0.5F},
                               {0.0F, 2.0F, 5.0F, 1.0F}},
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  auto next_tokens = sampler.sample(logits, args).cpu();

  EXPECT_TRUE(torch::equal(next_tokens, torch::tensor({1, 2}, torch::kInt32)));
}

}  // namespace
}  // namespace sglang
