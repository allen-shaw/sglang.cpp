#include <gtest/gtest.h>
#include <torch/torch.h>
#include "sglang/layers/activation.h"

TEST(ActivationTest, SiluAndMul) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    auto input = torch::randn({10, 256}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    auto out = torch::empty({10, 128}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    
    sglang::silu_and_mul(out, input);
    
    auto out_cpu = out.cpu();
    ASSERT_EQ(out_cpu.sizes(), std::vector<int64_t>({10, 128}));
}

TEST(ActivationTest, GeluAndMul) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    auto input = torch::randn({10, 256}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    auto out = torch::empty({10, 128}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    
    sglang::gelu_and_mul(out, input);
    
    auto out_cpu = out.cpu();
    ASSERT_EQ(out_cpu.sizes(), std::vector<int64_t>({10, 128}));
}
