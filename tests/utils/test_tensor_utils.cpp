#include <gtest/gtest.h>
#include "sglang/utils/tensor_utils.h"

namespace sglang {
namespace {

TEST(TensorUtilsTest, VecToTensorInt) {
  auto tensor = vec_to_tensor<int>({1, 2, 3, 4, 5});
  EXPECT_EQ(tensor.size(0), 5);
  EXPECT_EQ(tensor.dim(), 1);
  EXPECT_EQ(tensor[0].item<int>(), 1);
  EXPECT_EQ(tensor[4].item<int>(), 5);
}

TEST(TensorUtilsTest, VecToTensorFloat) {
  auto tensor = vec_to_tensor<float>({1.0f, 2.5f, 3.7f});
  EXPECT_EQ(tensor.size(0), 3);
  EXPECT_FLOAT_EQ(tensor[1].item<float>(), 2.5f);
}

TEST(TensorUtilsTest, MoveToCpu) {
  auto tensor = torch::tensor({1, 2, 3});
  auto cpu_tensor = move_to_cpu(tensor);
  EXPECT_TRUE(cpu_tensor.device().is_cpu());
  EXPECT_EQ(cpu_tensor.size(0), 3);
}

TEST(TensorUtilsTest, PrintTensorDoesNotCrash) {
  auto tensor = torch::tensor({1, 2, 3});
  // Just verify this doesn't segfault or throw
  EXPECT_NO_THROW(print_tensor(tensor, "test_tensor"));
}

TEST(TensorUtilsTest, PrintTensorLargeTensor) {
  auto tensor = torch::zeros({100});
  // Large tensor should not print content (only metadata)
  EXPECT_NO_THROW(print_tensor(tensor, "large_tensor"));
}

}  // namespace
}  // namespace sglang
