#include <gtest/gtest.h>
#include "sglang/core/sampling_params.h"

namespace sglang {
namespace {

TEST(SamplingParamsTest, DefaultIsNotGreedy) {
  SamplingParams params;
  // Default temperature=1.0, top_k=-1 → not greedy
  EXPECT_FALSE(params.is_greedy());
}

TEST(SamplingParamsTest, ZeroTemperatureIsGreedy) {
  SamplingParams params;
  params.temperature = 0.0f;
  EXPECT_TRUE(params.is_greedy());
}

TEST(SamplingParamsTest, NegativeTemperatureIsGreedy) {
  SamplingParams params;
  params.temperature = -1.0f;
  EXPECT_TRUE(params.is_greedy());
}

TEST(SamplingParamsTest, TopK1IsGreedy) {
  SamplingParams params;
  params.top_k = 1;
  EXPECT_TRUE(params.is_greedy());
}

TEST(SamplingParamsTest, HighTemperatureWithTopKIsNotGreedy) {
  SamplingParams params;
  params.temperature = 0.8f;
  params.top_k = 50;
  EXPECT_FALSE(params.is_greedy());
}

TEST(SamplingParamsTest, DefaultValues) {
  SamplingParams params;
  EXPECT_EQ(params.n, 1);
  EXPECT_EQ(params.max_new_tokens, 16);
  EXPECT_FLOAT_EQ(params.top_p, 1.0f);
  EXPECT_FLOAT_EQ(params.repetition_penalty, 1.0f);
  EXPECT_FALSE(params.ignore_eos);
  EXPECT_TRUE(params.skip_special_tokens);
}

}  // namespace
}  // namespace sglang
