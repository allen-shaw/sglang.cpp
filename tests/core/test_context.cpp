#include <gtest/gtest.h>
#include "sglang/core/context.h"

namespace sglang {
namespace {

// Reset global context between tests
class ContextTest : public ::testing::Test {
 protected:
  void TearDown() override {
    // The global context is static; we need to handle cleanup.
    // Since set_global_ctx throws if already set, we avoid setting it in
    // tests that don't need it and test it once.
  }
};

TEST(ContextTest, SetAndGetBatch) {
  auto ctx = std::make_shared<Context>(16, nullptr);

  EXPECT_EQ(ctx->get_batch(), nullptr);

  auto batch = std::make_shared<Batch>();
  batch->phase = BatchPhase::Prefill;

  ctx->set_batch(batch);
  EXPECT_NE(ctx->get_batch(), nullptr);
  EXPECT_TRUE(ctx->get_batch()->is_prefill());

  ctx->clear_batch();
  EXPECT_EQ(ctx->get_batch(), nullptr);
}

TEST(ContextTest, NestedSetBatchThrows) {
  auto ctx = std::make_shared<Context>(16, nullptr);
  auto batch1 = std::make_shared<Batch>();
  batch1->phase = BatchPhase::Prefill;
  auto batch2 = std::make_shared<Batch>();
  batch2->phase = BatchPhase::Decode;

  ctx->set_batch(batch1);
  EXPECT_THROW(ctx->set_batch(batch2), std::runtime_error);
  ctx->clear_batch();
}

TEST(ContextTest, BatchGuardRAII) {
  auto ctx = std::make_shared<Context>(16, nullptr);
  auto batch = std::make_shared<Batch>();
  batch->phase = BatchPhase::Decode;

  {
    BatchGuard guard(ctx, batch);
    EXPECT_NE(ctx->get_batch(), nullptr);
    EXPECT_TRUE(ctx->get_batch()->is_decode());
  }
  // After guard goes out of scope, batch should be cleared
  EXPECT_EQ(ctx->get_batch(), nullptr);
}

TEST(ContextTest, BatchGuardCleansUpOnException) {
  auto ctx = std::make_shared<Context>(16, nullptr);
  auto batch = std::make_shared<Batch>();
  batch->phase = BatchPhase::Prefill;

  try {
    BatchGuard guard(ctx, batch);
    EXPECT_NE(ctx->get_batch(), nullptr);
    throw std::runtime_error("test exception");
  } catch (const std::runtime_error&) {
    // Guard destructor should have cleared the batch
  }
  EXPECT_EQ(ctx->get_batch(), nullptr);
}

TEST(ContextTest, PageSizeDefault) {
  auto ctx = std::make_shared<Context>(32, nullptr);
  EXPECT_EQ(ctx->page_size, 32);
}

}  // namespace
}  // namespace sglang
