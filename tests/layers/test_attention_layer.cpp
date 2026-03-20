#include <gtest/gtest.h>
#include <torch/torch.h>
#include "sglang/layers/attention_layer.h"

using namespace sglang;

TEST(AttentionLayerTest, ExpectedThrowDueToIncompleteContext) {
    int layer_id = 0;
    int num_qo_heads = 32;
    int num_kv_heads = 8;
    int head_dim = 128;
    
    // Test instantiation
    AttentionLayer layer(layer_id, num_qo_heads, num_kv_heads, head_dim, nullptr, nullptr, nullptr);
    
    int total_tokens = 5;
    auto qkv = torch::randn({total_tokens, (num_qo_heads + 2 * num_kv_heads) * head_dim}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
    auto pos = torch::arange(total_tokens, torch::device(torch::kCUDA).dtype(torch::kInt32));
    
    // We expect a TORCH_CHECK failure since AttentionLayer relies on Phase 4 context wiring
    EXPECT_THROW(layer.forward(qkv, pos), c10::Error);
}
