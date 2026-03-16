#include <gtest/gtest.h>
#include "sglang/core/batch.h"

namespace sglang {
namespace {

TEST(BatchTest, IsPrefillAndDecode) {
  Batch batch;
  batch.phase = BatchPhase::Prefill;
  EXPECT_TRUE(batch.is_prefill());
  EXPECT_FALSE(batch.is_decode());

  batch.phase = BatchPhase::Decode;
  EXPECT_FALSE(batch.is_prefill());
  EXPECT_TRUE(batch.is_decode());
}

TEST(BatchTest, Size) {
  Batch batch;
  batch.phase = BatchPhase::Prefill;

  auto req1 = std::make_shared<Req>();
  req1->input_ids = torch::tensor({1, 2, 3}, torch::kInt32);
  auto req2 = std::make_shared<Req>();
  req2->input_ids = torch::tensor({4, 5}, torch::kInt32);

  batch.reqs.push_back(req1);
  batch.reqs.push_back(req2);

  EXPECT_EQ(batch.size(), 2);
}

TEST(BatchTest, PaddedSize) {
  Batch batch;
  batch.phase = BatchPhase::Decode;

  auto req1 = std::make_shared<Req>();
  req1->input_ids = torch::tensor({1}, torch::kInt32);
  auto req2 = std::make_shared<Req>();
  req2->input_ids = torch::tensor({2}, torch::kInt32);
  auto req3 = std::make_shared<Req>();
  req3->input_ids = torch::tensor({3}, torch::kInt32);

  batch.reqs.push_back(req1);
  batch.reqs.push_back(req2);

  // padded_reqs might have extra padding entries
  batch.padded_reqs.push_back(req1);
  batch.padded_reqs.push_back(req2);
  batch.padded_reqs.push_back(req3);

  EXPECT_EQ(batch.size(), 2);
  EXPECT_EQ(batch.padded_size(), 3);
}

}  // namespace
}  // namespace sglang
