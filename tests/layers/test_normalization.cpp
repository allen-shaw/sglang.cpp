#include <gtest/gtest.h>
#include <torch/torch.h>
#include "sglang/layers/normalization.h"

TEST(NormalizationTest, RMSNorm) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    
    int size = 128;
    int batch = 10;
    sglang::RMSNorm norm(size, 1e-5);
    
    auto input = torch::randn({batch, size}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    auto out = norm.forward(input);
    
    auto out_cpu = out.cpu();
    ASSERT_EQ(out_cpu.sizes(), std::vector<int64_t>({batch, size}));
}
