#include <gtest/gtest.h>
#include "sglang/message/message.h"

namespace sglang {
namespace {

TEST(MessageTest, GenerateRequestDefaultValues) {
  GenerateRequest req;
  EXPECT_EQ(req.uid, 0);
  EXPECT_FALSE(req.input_ids.defined());
  EXPECT_FLOAT_EQ(req.sampling_params.temperature, 1.0f);
}

TEST(MessageTest, GenerateRequestWithData) {
  GenerateRequest req;
  req.uid = 123;
  req.input_ids = torch::tensor({1, 2, 3, 4}, torch::kInt32);
  req.sampling_params.temperature = 0.0f;
  req.sampling_params.max_new_tokens = 100;

  EXPECT_EQ(req.uid, 123);
  EXPECT_EQ(req.input_ids.size(0), 4);
  EXPECT_TRUE(req.sampling_params.is_greedy());
  EXPECT_EQ(req.sampling_params.max_new_tokens, 100);
}

TEST(MessageTest, GenerateResponseDefaultValues) {
  GenerateResponse resp;
  EXPECT_EQ(resp.uid, 0);
  EXPECT_TRUE(resp.incremental_output.empty());
  EXPECT_FALSE(resp.finished);
}

TEST(MessageTest, GenerateResponseWithData) {
  GenerateResponse resp;
  resp.uid = 42;
  resp.incremental_output = "Hello, world!";
  resp.finished = true;

  EXPECT_EQ(resp.uid, 42);
  EXPECT_EQ(resp.incremental_output, "Hello, world!");
  EXPECT_TRUE(resp.finished);
}

TEST(MessageTest, BatchResponseAccumulation) {
  BatchResponse batch;
  batch.data.push_back({1, "token1", false});
  batch.data.push_back({2, "token2", false});
  batch.data.push_back({1, " token3", true});

  EXPECT_EQ(batch.data.size(), 3);
  EXPECT_EQ(batch.data[0].uid, 1);
  EXPECT_TRUE(batch.data[2].finished);
}

TEST(MessageTest, ExitAndAbortMessages) {
  ExitMessage exit;
  // ExitMessage is just a signal, no data

  AbortMessage abort;
  abort.uid = 99;
  EXPECT_EQ(abort.uid, 99);
}

}  // namespace
}  // namespace sglang
