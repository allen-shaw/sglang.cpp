#include <gtest/gtest.h>
#include "sglang/core/req.h"

namespace sglang {
namespace {

TEST(ReqTest, BasicProperties) {
  Req req;
  req.req_id = 42;
  req.input_ids = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
  req.table_idx = 0;
  req.cached_len = 0;
  req.output_len = 3;

  EXPECT_EQ(req.device_len(), 5);
  EXPECT_EQ(req.max_device_len(), 8);  // 5 + 3
  EXPECT_EQ(req.remain_len(), 3);
  EXPECT_EQ(req.extend_len(), 5);  // 5 - 0
  EXPECT_TRUE(req.can_decode());
}

TEST(ReqTest, CompleteOneAdvancesDeviceBeforeHostAppend) {
  Req req;
  req.input_ids = torch::tensor({1, 2, 3}, torch::kInt32);
  req.cached_len = 0;
  req.output_len = 2;
  req.initialize_runtime_state();

  req.complete_one();
  EXPECT_EQ(req.cached_len, 3);
  EXPECT_EQ(req.device_len(), 4);
  EXPECT_EQ(req.remain_len(), 1);

  auto next_token = torch::tensor({10}, torch::kInt32);
  req.append_host(next_token);
  EXPECT_EQ(req.device_len(), 4);     // 3 + 1 appended
  EXPECT_EQ(req.cached_len, 3);       // append_host does not change cached_len
  EXPECT_EQ(req.remain_len(), 1);
  EXPECT_EQ(req.extend_len(), 1);     // 4 - 3
  EXPECT_TRUE(req.can_decode());
}

TEST(ReqTest, CanDecodeReturnsFalseWhenDone) {
  Req req;
  req.input_ids = torch::tensor({1}, torch::kInt32);
  req.cached_len = 0;
  req.output_len = 1;
  req.initialize_runtime_state();

  req.complete_one();
  req.append_host(torch::tensor({2}, torch::kInt32));

  EXPECT_FALSE(req.can_decode());
}

TEST(ReqTest, AppendHostRequiresAdvancedDeviceLen) {
  Req req;
  req.input_ids = torch::tensor({1, 2, 3}, torch::kInt32);
  req.output_len = 1;
  req.initialize_runtime_state();

  EXPECT_THROW(req.append_host(torch::tensor({4}, torch::kInt32)), c10::Error);
  req.complete_one();
  req.append_host(torch::tensor({4}, torch::kInt32));
  EXPECT_EQ(req.input_ids.size(0), 4);
  EXPECT_EQ(req.input_ids[3].item<int>(), 4);
}

TEST(ReqTest, ChunkedPrefillNeverDecodes) {
  Req req;
  req.input_ids = torch::tensor({1, 2, 3}, torch::kInt32);
  req.cached_len = 1;
  req.output_len = 4;
  req.is_chunked_prefill = true;
  req.initialize_runtime_state();

  EXPECT_FALSE(req.can_decode());
}

TEST(ReqTest, ToString) {
  Req req;
  req.input_ids = torch::tensor({1, 2, 3}, torch::kInt32);
  req.table_idx = 5;
  req.cached_len = 1;
  req.output_len = 10;

  std::string s = req.toString();
  EXPECT_NE(s.find("table_idx=5"), std::string::npos);
  EXPECT_NE(s.find("cached_len=1"), std::string::npos);
  EXPECT_NE(s.find("device_len=3"), std::string::npos);
  EXPECT_NE(s.find("max_device_len=13"), std::string::npos);
}

}  // namespace
}  // namespace sglang
