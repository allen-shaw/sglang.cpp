#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cstdlib>
#include <filesystem>
#include "sglang/models/qwen3_moe.h"
#include "sglang/models/weight_loader.h"

using namespace sglang;

namespace {

std::string find_qwen3_moe_model_path() {
    const char* env_path = std::getenv("QWEN3_MOE_MODEL_PATH");
    if (env_path && std::filesystem::exists(std::string(env_path) + "/config.json")) {
        return std::string(env_path);
    }

    const std::string downloaded = "/root/workspace/models/tiny-Qwen3MoeForCausalLM";
    if (std::filesystem::exists(downloaded + "/config.json")) {
        return downloaded;
    }

    return "";
}

}  // namespace

TEST(Qwen3MoETest, FullModelInitializationRegistersMoEWeights) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required because model linear weights are CUDA tensors";
    }

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
    config.tie_word_embeddings = true;
    config.num_experts = 8;
    config.num_experts_per_tok = 2;
    config.moe_intermediate_size = 128;
    config.norm_topk_prob = false;
    config.model_type = "qwen3_moe";
    config.rotary_config = {32, 32, 1024, 10000.0f};

    Qwen3MoEForCausalLM model(config);

    auto params = model.named_parameters();
    ASSERT_TRUE(params.contains("model.layers_0.mlp.experts.gate_up_proj"));
    ASSERT_TRUE(params.contains("model.layers_0.mlp.experts.down_proj"));
    ASSERT_TRUE(params.contains("model.layers_0.self_attn.q_norm.weight"));
    ASSERT_TRUE(params.contains("model.layers_0.self_attn.k_norm.weight"));
    ASSERT_FALSE(params.contains("model.layers_0.self_attn.qkv_proj.bias"));
    EXPECT_EQ(
        params["model.layers_0.mlp.experts.gate_up_proj"].sizes().vec(),
        std::vector<int64_t>({8, 256, 128}));
    EXPECT_EQ(
        params["model.layers_0.mlp.experts.down_proj"].sizes().vec(),
        std::vector<int64_t>({8, 128, 128}));
}

TEST(Qwen3MoETest, TinyRealCheckpointLoadsWeights) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required for real Qwen3-MOE checkpoint load test";
    }

    const auto model_path = find_qwen3_moe_model_path();
    if (model_path.empty()) {
        GTEST_SKIP() << "Qwen3-MOE model path not found. Set QWEN3_MOE_MODEL_PATH";
    }

    auto config = ModelConfig::from_json_file(model_path + "/config.json");
    EXPECT_EQ(config.model_type, "qwen3_moe");
    ASSERT_TRUE(config.is_moe());
    ASSERT_GT(config.num_experts, 0);
    ASSERT_GT(config.num_experts_per_tok, 0);
    ASSERT_GT(config.moe_intermediate_size, 0);

    Qwen3MoEForCausalLM model(config);
    ASSERT_NO_THROW(WeightLoader::load_weights(model, model_path, torch::kBFloat16, torch::kCUDA));

    auto params = model.named_parameters();
    ASSERT_TRUE(params.contains("model.layers_0.mlp.experts.gate_up_proj"));
    ASSERT_TRUE(params.contains("model.layers_0.mlp.experts.down_proj"));
    EXPECT_EQ(
        params["model.layers_0.mlp.experts.gate_up_proj"].sizes().vec(),
        std::vector<int64_t>({config.num_experts, config.moe_intermediate_size * 2, config.hidden_size}));
    EXPECT_EQ(
        params["model.layers_0.mlp.experts.down_proj"].sizes().vec(),
        std::vector<int64_t>({config.num_experts, config.hidden_size, config.moe_intermediate_size}));
    EXPECT_GT(params["model.layers_0.mlp.experts.gate_up_proj"].abs().sum().item<float>(), 0.0F);
    EXPECT_GT(params["model.layers_0.mlp.experts.down_proj"].abs().sum().item<float>(), 0.0F);
}
