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

TEST(NormalizationTest, FusedQKRMSNormMatchesSeparateKernels) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available, skipping test.";
    }

    constexpr int tokens = 7;
    constexpr int q_heads = 16;
    constexpr int k_heads = 8;
    constexpr int head_dim = 128;

    sglang::RMSNorm q_norm(head_dim, 1e-6f);
    sglang::RMSNorm k_norm(head_dim, 1e-6f);
    torch::NoGradGuard no_grad;
    q_norm.weight.copy_(torch::randn_like(q_norm.weight));
    k_norm.weight.copy_(torch::randn_like(k_norm.weight));

    auto q = torch::randn(
        {tokens, q_heads, head_dim},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
    auto k = torch::randn(
        {tokens, k_heads, head_dim},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
    auto q_expected = q.clone();
    auto k_expected = k.clone();

    q_norm.forward_inplace_3d_strided(q_expected);
    k_norm.forward_inplace_3d_strided(k_expected);
    sglang::fused_qk_rmsnorm_inplace_3d_strided(q, k, q_norm, k_norm);
    torch::cuda::synchronize();

    EXPECT_TRUE(torch::allclose(q, q_expected, /*rtol=*/1e-3, /*atol=*/1e-3));
    EXPECT_TRUE(torch::allclose(k, k_expected, /*rtol=*/1e-3, /*atol=*/1e-3));
}
