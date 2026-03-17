#include <gtest/gtest.h>
#include "sglang/kernels/radix.h"

namespace sglang {
namespace {

TEST(FastCompareKeyTest, IdenticalTensors) {
    auto tensor = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
    EXPECT_EQ(fast_compare_key(tensor, tensor), 5);
}

TEST(FastCompareKeyTest, FullyDifferent) {
    auto lhs = torch::tensor({1, 2, 3}, torch::kInt32);
    auto rhs = torch::tensor({9, 8, 7}, torch::kInt32);
    EXPECT_EQ(fast_compare_key(lhs, rhs), 0);
}

TEST(FastCompareKeyTest, PartialMatch) {
    auto lhs = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
    auto rhs = torch::tensor({1, 2, 3, 9, 9}, torch::kInt32);
    EXPECT_EQ(fast_compare_key(lhs, rhs), 3);
}

TEST(FastCompareKeyTest, EmptyTensors) {
    auto empty = torch::empty({0}, torch::kInt32);
    EXPECT_EQ(fast_compare_key(empty, empty), 0);
}

TEST(FastCompareKeyTest, DifferentLengths) {
    auto lhs = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
    auto rhs = torch::tensor({1, 2, 3}, torch::kInt32);
    EXPECT_EQ(fast_compare_key(lhs, rhs), 3);
}

TEST(FastCompareKeyTest, DifferentLengthsPartialMatch) {
    auto lhs = torch::tensor({1, 2, 9}, torch::kInt32);
    auto rhs = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32);
    EXPECT_EQ(fast_compare_key(lhs, rhs), 2);
}

TEST(FastCompareKeyTest, SingleElement) {
    auto lhs = torch::tensor({42}, torch::kInt32);
    auto rhs = torch::tensor({42}, torch::kInt32);
    EXPECT_EQ(fast_compare_key(lhs, rhs), 1);

    auto rhs2 = torch::tensor({99}, torch::kInt32);
    EXPECT_EQ(fast_compare_key(lhs, rhs2), 0);
}

}  // namespace
}  // namespace sglang
