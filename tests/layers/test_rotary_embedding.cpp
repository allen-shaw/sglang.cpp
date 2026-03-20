#include <gtest/gtest.h>
#include <torch/torch.h>
#include "sglang/layers/rotary_embedding.h"

TEST(RotaryEmbeddingTest, ForwardInplace) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    
    int batch_size = 2;
    int seq_len = 5;
    int num_qo_heads = 8;
    int num_kv_heads = 2;
    int head_dim = 64;
    int rotary_dim = 64;
    int max_pos = 100;
    
    sglang::RotaryEmbedding rope(head_dim, rotary_dim, max_pos, 10000.0);
    
    // positions: [10]
    auto positions = torch::arange(0, batch_size * seq_len, torch::device(torch::kCUDA).dtype(torch::kInt32));
    
    // query: [10, 8, 64], key: [10, 2, 64]
    auto query = torch::randn({batch_size * seq_len, num_qo_heads, head_dim}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    auto key = torch::randn({batch_size * seq_len, num_kv_heads, head_dim}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    
    // cos_cache, sin_cache: [100, 64]
    auto cos_cache = torch::randn({max_pos, rotary_dim}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto sin_cache = torch::randn({max_pos, rotary_dim}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    
    rope.forward_inplace(positions, query, key, cos_cache, sin_cache);
    
    auto q_cpu = query.cpu();
    ASSERT_EQ(q_cpu.sizes(), std::vector<int64_t>({batch_size * seq_len, num_qo_heads, head_dim}));
}
