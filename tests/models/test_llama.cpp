#include <gtest/gtest.h>
#include <torch/torch.h>
#include "sglang/models/llama.h"

using namespace sglang;

TEST(LlamaTest, DecoderLayerInitialization) {
    ModelConfig config;
    config.hidden_size = 128;
    config.intermediate_size = 256;
    config.num_qo_heads = 4;
    config.num_kv_heads = 4;
    config.head_dim = 32;
    config.rms_norm_eps = 1e-5;
    config.hidden_act = "silu";
    config.rotary_config = {32, 32, 1024, 10000.0f};
    
    LlamaDecoderLayer layer(config, 0);

    // Mock tensor inputs
    auto x = torch::randn({2, 128});
    auto pos = torch::tensor({0, 1}, torch::kInt64);

    // Run forward pass placeholder test
    // Assuming Linear implementation doesn't fail inherently for shape {2, 128} -> QKV (2, 128)
    auto out = layer.forward(x, pos);
    EXPECT_EQ(out.sizes().vec(), std::vector<int64_t>({2, 128}));
}

TEST(LlamaTest, FullModelInitialization) {
    ModelConfig config;
    config.hidden_size = 128;
    config.vocab_size = 1000;
    config.num_layers = 2;
    config.intermediate_size = 256;
    config.num_qo_heads = 4;
    config.num_kv_heads = 4;
    config.head_dim = 32;
    config.rms_norm_eps = 1e-5;
    config.hidden_act = "silu";
    config.tie_word_embeddings = false;
    config.rotary_config = {32, 32, 1024, 10000.0f};

    LlamaForCausalLM model(config);

    auto input_ids = torch::tensor({10, 20}, torch::kInt64);
    auto pos = torch::tensor({0, 1}, torch::kInt64);

    auto logits = model.forward(input_ids, pos);
    EXPECT_EQ(logits.sizes().vec(), std::vector<int64_t>({2, 1000}));
}
