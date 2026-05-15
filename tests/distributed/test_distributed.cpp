#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cstdlib>

#include "sglang/distributed/distributed.h"

namespace sglang {
namespace {

int local_rank() {
  if (const char* value = std::getenv("SGLANG_TP_LOCAL_RANK")) {
    return std::stoi(value);
  }
  if (const char* value = std::getenv("LOCAL_RANK")) {
    return std::stoi(value);
  }
  return tp_rank();
}

}  // namespace

TEST(DistributedTest, TensorParallelAllReduce) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is required for NCCL tensor-parallel all-reduce";
  }

  auto info = get_tp_info();
  if (!info.enabled()) {
    GTEST_SKIP() << "Set SGLANG_TP_SIZE>1 to run NCCL tensor-parallel all-reduce";
  }

  auto device = torch::Device(torch::kCUDA, local_rank());
  auto x = torch::full({8}, static_cast<float>(info.rank + 1),
                       torch::TensorOptions().device(device).dtype(torch::kFloat32));
  auto y = tensor_model_parallel_all_reduce(x);
  torch::cuda::synchronize(device.index());

  const float expected = static_cast<float>(info.size * (info.size + 1) / 2);
  EXPECT_TRUE(torch::allclose(y.cpu(), torch::full({8}, expected)));
}

}  // namespace sglang
