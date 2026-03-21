#include <gtest/gtest.h>
#include <torch/torch.h>

#include "sglang/models/config.h"
#include "sglang/models/llama.h"
#include "sglang/models/qwen.h"
#include "sglang/moe/moe.h"

using namespace sglang;

ModelConfig generate_small_config() {
    ModelConfig config;
    config.num_layers = 1;
    config.num_qo_heads = 4;
    config.num_kv_heads = 2;
    config.head_dim = 16;
    config.hidden_size = 64;
    config.vocab_size = 256;
    config.intermediate_size = 128;
    config.rms_norm_eps = 1e-5f;
    config.hidden_act = "silu";
    config.tie_word_embeddings = false;
    
    config.rotary_config.head_dim = 16;
    config.rotary_config.rotary_dim = 16;
    config.rotary_config.max_position = 1024;
    config.rotary_config.base = 10000.0f;
    
    config.num_experts = 4;
    config.num_experts_per_tok = 2;
    config.moe_intermediate_size = 128;
    return config;
}

TEST(ModelsTest, LlamaForCausalLM_Initialization) {
    auto config = generate_small_config();
    LlamaForCausalLMImpl model(config);
    // Sanity check to make sure parameters are registered
    // A fully registered module inside Torch should have non-zero parameters
    auto params = model.parameters();
    EXPECT_GT(params.size(), 0);
}

TEST(ModelsTest, LlamaForCausalLM_ForwardShape) {
    auto config = generate_small_config();
    LlamaForCausalLMImpl model(config);

    torch::Tensor input_ids = torch::randint(0, config.vocab_size, {2, 10}, torch::kInt32); // [batch=2, seq=10]
    torch::Tensor positions = torch::arange(0, 10, torch::kInt32).unsqueeze(0).expand({2, 10}).contiguous();

    auto logits = model.forward(input_ids, positions);
    
    // Expect shape [2, 10, vocab_size]
    EXPECT_EQ(logits.dim(), 3);
    EXPECT_EQ(logits.size(0), 2);
    EXPECT_EQ(logits.size(1), 10);
    EXPECT_EQ(logits.size(2), config.vocab_size);
}

TEST(ModelsTest, Qwen2ForCausalLM_ForwardShape) {
    auto config = generate_small_config();
    Qwen2ForCausalLMImpl model(config);

    torch::Tensor input_ids = torch::randint(0, config.vocab_size, {1, 5}, torch::kInt32);
    torch::Tensor positions = torch::arange(0, 5, torch::kInt32).unsqueeze(0);

    auto logits = model.forward(input_ids, positions);
    
    EXPECT_EQ(logits.dim(), 3);
    EXPECT_EQ(logits.size(0), 1);
    EXPECT_EQ(logits.size(1), 5);
    EXPECT_EQ(logits.size(2), config.vocab_size);
}

TEST(ModelsTest, Qwen3MoeForCausalLM_ForwardShape) {
    auto config = generate_small_config();
    Qwen3MoeForCausalLMImpl model(config);

    torch::Tensor input_ids = torch::randint(0, config.vocab_size, {3, 8}, torch::kInt32);
    torch::Tensor positions = torch::arange(0, 8, torch::kInt32).unsqueeze(0).expand({3, 8}).contiguous();

    auto logits = model.forward(input_ids, positions);
    
    EXPECT_EQ(logits.dim(), 3);
    EXPECT_EQ(logits.size(0), 3);
    EXPECT_EQ(logits.size(1), 8);
    EXPECT_EQ(logits.size(2), config.vocab_size);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
