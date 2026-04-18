#include <gtest/gtest.h>
#include "sglang/kvcache/cache_manager.h"

namespace sglang {
namespace {

class RadixPrefixCacheTest : public ::testing::Test {
 protected:
    // Use page_size = 1 for simplicity
    RadixPrefixCache cache{1};
};

TEST_F(RadixPrefixCacheTest, EmptyMatchPrefix) {
    auto input = torch::tensor({1, 2, 3}, torch::kInt32);
    auto result = cache.match_prefix(input);
    EXPECT_EQ(result.handle->cached_len, 0);

    auto indices = result.handle->get_matched_indices();
    EXPECT_EQ(indices.size(0), 0);
}

TEST_F(RadixPrefixCacheTest, InsertAndMatch) {
    auto input_ids = torch::tensor({10, 20, 30, 40}, torch::kInt32);
    auto indices = torch::tensor({0, 1, 2, 3}, torch::kInt32);

    auto insert_result = cache.insert_prefix(input_ids, indices);
    EXPECT_EQ(insert_result.cached_len, 0);
    EXPECT_EQ(insert_result.handle->cached_len, 4);

    // Now match the same prefix
    auto match_result = cache.match_prefix(input_ids);
    EXPECT_EQ(match_result.handle->cached_len, 4);

    auto matched_indices = match_result.handle->get_matched_indices();
    EXPECT_EQ(matched_indices.size(0), 4);
    EXPECT_TRUE(torch::equal(matched_indices, indices));
}

TEST_F(RadixPrefixCacheTest, PartialMatch) {
    auto input_ids = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
    auto indices = torch::tensor({10, 20, 30, 40, 50}, torch::kInt32);
    cache.insert_prefix(input_ids, indices);

    // Match with a longer input that shares a prefix
    auto query = torch::tensor({1, 2, 3, 9, 9}, torch::kInt32);
    auto match_result = cache.match_prefix(query);
    EXPECT_EQ(match_result.handle->cached_len, 3);
}

TEST_F(RadixPrefixCacheTest, SizeInfoAfterInsert) {
    auto ids = torch::tensor({1, 2, 3}, torch::kInt32);
    auto idx = torch::tensor({0, 1, 2}, torch::kInt32);

    auto info_before = cache.size_info();
    EXPECT_EQ(info_before.evictable_size, 0);
    EXPECT_EQ(info_before.protected_size, 0);

    cache.insert_prefix(ids, idx);

    auto info_after = cache.size_info();
    EXPECT_EQ(info_after.evictable_size, 3);
    EXPECT_EQ(info_after.protected_size, 0);
}

TEST_F(RadixPrefixCacheTest, LockAndUnlock) {
    auto ids = torch::tensor({1, 2, 3}, torch::kInt32);
    auto idx = torch::tensor({0, 1, 2}, torch::kInt32);

    auto result = cache.insert_prefix(ids, idx);
    EXPECT_EQ(cache.size_info().evictable_size, 3);

    // Lock: moves from evictable to protected
    cache.lock_handle(result.handle, false);
    EXPECT_EQ(cache.size_info().evictable_size, 0);
    EXPECT_EQ(cache.size_info().protected_size, 3);

    // Unlock: moves back to evictable
    cache.lock_handle(result.handle, true);
    EXPECT_EQ(cache.size_info().evictable_size, 3);
    EXPECT_EQ(cache.size_info().protected_size, 0);
}

TEST_F(RadixPrefixCacheTest, EvictBasic) {
    auto ids = torch::tensor({1, 2, 3}, torch::kInt32);
    auto idx = torch::tensor({0, 1, 2}, torch::kInt32);

    cache.insert_prefix(ids, idx);
    EXPECT_EQ(cache.size_info().evictable_size, 3);

    auto evicted = cache.evict(2);
    // Should evict the leaf node (all 3 indices, since it's one node)
    EXPECT_GE(evicted.size(0), 2);
}

TEST_F(RadixPrefixCacheTest, EvictZero) {
    auto evicted = cache.evict(0);
    EXPECT_EQ(evicted.size(0), 0);
}

TEST_F(RadixPrefixCacheTest, EvictProtectedFails) {
    auto ids = torch::tensor({1, 2, 3}, torch::kInt32);
    auto idx = torch::tensor({0, 1, 2}, torch::kInt32);

    auto result = cache.insert_prefix(ids, idx);
    cache.lock_handle(result.handle, false);  // Lock it

    EXPECT_THROW(cache.evict(1), c10::Error);

    // Cleanup: unlock to avoid issues
    cache.lock_handle(result.handle, true);
}

TEST_F(RadixPrefixCacheTest, InsertMatchEvictCycle) {
    // Insert two different prefixes
    cache.insert_prefix(torch::tensor({1, 2, 3}, torch::kInt32),
                        torch::tensor({10, 11, 12}, torch::kInt32));
    cache.insert_prefix(torch::tensor({4, 5, 6}, torch::kInt32),
                        torch::tensor({20, 21, 22}, torch::kInt32));

    EXPECT_EQ(cache.size_info().evictable_size, 6);

    // Evict one prefix worth
    auto evicted = cache.evict(3);
    EXPECT_EQ(evicted.size(0), 3);
    EXPECT_EQ(cache.size_info().evictable_size, 3);

    // The remaining prefix should still be matchable
    // (one of them was evicted, so we check both)
    auto match1 = cache.match_prefix(torch::tensor({1, 2, 3}, torch::kInt32));
    auto match2 = cache.match_prefix(torch::tensor({4, 5, 6}, torch::kInt32));

    // Exactly one should still be cached
    EXPECT_TRUE(match1.handle->cached_len == 3 ||
                match2.handle->cached_len == 3);
}

TEST_F(RadixPrefixCacheTest, DuplicateInsert) {
    auto ids = torch::tensor({1, 2, 3}, torch::kInt32);
    auto idx = torch::tensor({0, 1, 2}, torch::kInt32);

    cache.insert_prefix(ids, idx);
    // Insert the same prefix again — should not create additional nodes
    auto result = cache.insert_prefix(ids, idx);
    EXPECT_EQ(result.cached_len, 3);  // already cached
    EXPECT_EQ(cache.size_info().evictable_size, 3);  // size unchanged
}

}  // namespace
}  // namespace sglang
