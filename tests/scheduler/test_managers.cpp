#include <gtest/gtest.h>

#include "sglang/scheduler/decode.h"
#include "sglang/scheduler/prefill.h"

namespace sglang {
namespace {

std::shared_ptr<Req> make_decode_req(uint64_t req_id, int input_len, int max_new_tokens) {
  auto req = std::make_shared<Req>();
  req->req_id = req_id;
  req->input_ids = torch::arange(0, input_len, torch::TensorOptions().dtype(torch::kInt32));
  req->cached_len = input_len;
  req->output_len = max_new_tokens;
  req->device_len_ = input_len;
  req->max_device_len_ = input_len + max_new_tokens;
  req->validate_runtime_state();
  return req;
}

TEST(TableManagerTest, AllocateAndFreeSlots) {
  auto page_table = torch::zeros({3, 8}, torch::TensorOptions().dtype(torch::kInt32));
  TableManager table_manager(2, page_table);

  const int first = table_manager.allocate();
  const int second = table_manager.allocate();
  EXPECT_NE(first, second);
  EXPECT_EQ(table_manager.available_size(), 0);

  table_manager.free(first);
  EXPECT_EQ(table_manager.available_size(), 1);
}

TEST(DecodeManagerTest, FiltersOnlyDecodableReqs) {
  DecodeManager decode_manager(/*page_size=*/1);
  auto live_req = make_decode_req(1, 4, 2);
  auto finished_req = make_decode_req(2, 4, 0);

  decode_manager.filter_reqs({live_req, finished_req});
  auto batch = decode_manager.schedule_next_batch();

  ASSERT_NE(batch, nullptr);
  ASSERT_EQ(batch->reqs.size(), 1);
  EXPECT_EQ(batch->reqs[0]->req_id, 1);
  EXPECT_EQ(decode_manager.inflight_tokens(), 2);
}

TEST(PrefillManagerTest, ChunksRequestWhenBudgetIsTooSmall) {
  auto page_table = torch::zeros({4, 16}, torch::TensorOptions().dtype(torch::kInt32));
  TableManager table_manager(3, page_table);
  CacheManager cache_manager(/*num_pages=*/32, /*page_size=*/1, page_table, "radix");
  DecodeManager decode_manager(/*page_size=*/1);
  PrefillManager prefill_manager(cache_manager, table_manager, decode_manager);

  GenerateRequest request;
  request.uid = 7;
  request.input_ids = torch::tensor({10, 11, 12, 13}, torch::kInt32);
  request.sampling_params.max_new_tokens = 3;
  prefill_manager.add_one_req(request);

  auto batch = prefill_manager.schedule_next_batch(/*prefill_budget=*/2);

  ASSERT_NE(batch, nullptr);
  ASSERT_EQ(batch->reqs.size(), 1);
  EXPECT_TRUE(batch->reqs[0]->is_chunked_prefill);
  EXPECT_EQ(batch->reqs[0]->input_ids.size(0), 2);
  EXPECT_TRUE(prefill_manager.runnable());
}

}  // namespace
}  // namespace sglang
