#include <gtest/gtest.h>
#include <torch/torch.h>
#include "sglang/distributed/distributed.h"
#include "sglang/moe/moe.h"
#include "sglang/models/weight_loader.h"

using namespace sglang;

namespace {

torch::Tensor reference_moe(const torch::Tensor& hidden,
                            const torch::Tensor& router_logits,
                            const torch::Tensor& gate_up_proj,
                            const torch::Tensor& down_proj,
                            int top_k,
                            bool renormalize) {
    const auto num_experts = gate_up_proj.size(0);
    const auto intermediate_size = gate_up_proj.size(1) / 2;

    auto routing_probs = torch::softmax(router_logits.to(torch::kFloat32), -1);
    auto topk = routing_probs.topk(top_k, -1);
    auto topk_weights = std::get<0>(topk);
    auto topk_ids = std::get<1>(topk).to(torch::kLong);
    if (renormalize) {
        topk_weights = topk_weights / (topk_weights.sum(-1, true) + 1e-8);
    }

    auto output = torch::zeros_like(hidden);
    for (int k = 0; k < top_k; ++k) {
        auto ids = topk_ids.select(1, k);
        auto weights = topk_weights.select(1, k).to(hidden.scalar_type());
        for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
            auto mask = ids.eq(expert_id);
            if (!mask.any().item<bool>()) {
                continue;
            }
            auto token_indices = torch::nonzero(mask).squeeze(1);
            auto expert_input = hidden.index_select(0, token_indices);
            auto gate_up = torch::nn::functional::linear(
                expert_input,
                gate_up_proj.select(0, expert_id),
                torch::Tensor());
            auto gate = gate_up.narrow(1, 0, intermediate_size);
            auto up = gate_up.narrow(1, intermediate_size, intermediate_size);
            auto activated = torch::silu(gate) * up;
            auto expert_output = torch::nn::functional::linear(
                activated,
                down_proj.select(0, expert_id),
                torch::Tensor());
            expert_output = expert_output * weights.index_select(0, token_indices).unsqueeze(1);
            output.index_add_(0, token_indices, expert_output);
        }
    }
    return output;
}

}  // namespace

TEST(MoETest, RegistersPackedExpertWeights) {
    MoELayer moe(4, 2, 8, 16, false);

    auto params = moe.named_parameters();
    ASSERT_TRUE(params.contains("gate_up_proj"));
    ASSERT_TRUE(params.contains("down_proj"));
    EXPECT_EQ(params["gate_up_proj"].sizes().vec(), std::vector<int64_t>({4, 32, 8}));
    EXPECT_EQ(params["down_proj"].sizes().vec(), std::vector<int64_t>({4, 8, 16}));
}

TEST(MoETest, ForwardMatchesReferenceTop2Renormalized) {
    torch::manual_seed(0);
    torch::NoGradGuard no_grad;

    constexpr int kNumExperts = 3;
    constexpr int kTopK = 2;
    constexpr int kHiddenSize = 4;
    constexpr int kIntermediateSize = 5;

    MoELayer moe(kNumExperts, kTopK, kHiddenSize, kIntermediateSize, true);
    auto hidden = torch::randn({6, kHiddenSize}, torch::kFloat32);
    auto router_logits = torch::tensor({
        {5.0F, 1.0F, -2.0F},
        {0.1F, 3.0F, 2.0F},
        {-1.0F, 0.0F, 4.0F},
        {2.0F, 2.5F, 0.0F},
        {0.0F, -0.5F, 1.5F},
        {3.0F, 0.0F, 3.5F},
    });

    moe.gate_up_proj.copy_(torch::randn_like(moe.gate_up_proj));
    moe.down_proj.copy_(torch::randn_like(moe.down_proj));

    auto out = moe.forward(hidden, router_logits);
    auto expected = reference_moe(
        hidden, router_logits, moe.gate_up_proj, moe.down_proj, kTopK, true);

    EXPECT_EQ(out.sizes().vec(), std::vector<int64_t>({6, kHiddenSize}));
    EXPECT_TRUE(torch::allclose(out, expected, /*rtol=*/1e-5, /*atol=*/1e-5));
    EXPECT_FALSE(torch::allclose(out, hidden, /*rtol=*/1e-5, /*atol=*/1e-5));
}

TEST(MoETest, WeightLoaderStacksExpertWeights) {
    std::unordered_map<std::string, torch::Tensor> raw;
    raw["model.layers.0.mlp.experts.0.gate_proj.weight"] = torch::full({2, 3}, 1.0F);
    raw["model.layers.0.mlp.experts.0.up_proj.weight"] = torch::full({2, 3}, 2.0F);
    raw["model.layers.0.mlp.experts.0.down_proj.weight"] = torch::full({3, 2}, 3.0F);
    raw["model.layers.0.mlp.experts.1.gate_proj.weight"] = torch::full({2, 3}, 4.0F);
    raw["model.layers.0.mlp.experts.1.up_proj.weight"] = torch::full({2, 3}, 5.0F);
    raw["model.layers.0.mlp.experts.1.down_proj.weight"] = torch::full({3, 2}, 6.0F);

    auto merged = WeightLoader::merge_weights(std::move(raw));

    ASSERT_TRUE(merged.count("model.layers.0.mlp.experts.gate_up_proj"));
    ASSERT_TRUE(merged.count("model.layers.0.mlp.experts.down_proj"));

    auto gate_up = merged.at("model.layers.0.mlp.experts.gate_up_proj");
    auto down = merged.at("model.layers.0.mlp.experts.down_proj");
    EXPECT_EQ(gate_up.sizes().vec(), std::vector<int64_t>({2, 4, 3}));
    EXPECT_EQ(down.sizes().vec(), std::vector<int64_t>({2, 3, 2}));
    EXPECT_TRUE(torch::allclose(gate_up[0].narrow(0, 0, 2), torch::full({2, 3}, 1.0F)));
    EXPECT_TRUE(torch::allclose(gate_up[0].narrow(0, 2, 2), torch::full({2, 3}, 2.0F)));
    EXPECT_TRUE(torch::allclose(gate_up[1].narrow(0, 0, 2), torch::full({2, 3}, 4.0F)));
    EXPECT_TRUE(torch::allclose(gate_up[1].narrow(0, 2, 2), torch::full({2, 3}, 5.0F)));
    EXPECT_TRUE(torch::allclose(down[0], torch::full({3, 2}, 3.0F)));
    EXPECT_TRUE(torch::allclose(down[1], torch::full({3, 2}, 6.0F)));
}

TEST(MoETest, TensorParallelExpertShapes) {
    set_tp_info_for_test(/*rank=*/1, /*size=*/2);

    MoELayer moe(/*num_experts=*/4, /*top_k=*/2, /*hidden_size=*/8, /*intermediate_size=*/16, false);
    auto params = moe.named_parameters();
    ASSERT_TRUE(params.contains("gate_up_proj"));
    ASSERT_TRUE(params.contains("down_proj"));
    EXPECT_EQ(params["gate_up_proj"].sizes().vec(), std::vector<int64_t>({4, 16, 8}));
    EXPECT_EQ(params["down_proj"].sizes().vec(), std::vector<int64_t>({4, 8, 8}));

    reset_tp_info_for_test();
}

TEST(MoETest, WeightLoaderShardsExpertWeightsForTensorParallel) {
    auto gate_up = torch::arange(4 * 32 * 8, torch::kFloat32).view({4, 32, 8});
    auto down = torch::arange(4 * 8 * 16, torch::kFloat32).view({4, 8, 16});

    auto gate_up_shard = WeightLoader::shard_tensor_for_parameter(
        "model.layers.0.mlp.experts.gate_up_proj",
        gate_up,
        torch::IntArrayRef({4, 16, 8}),
        /*tp_rank=*/1,
        /*tp_size=*/2);
    auto down_shard = WeightLoader::shard_tensor_for_parameter(
        "model.layers.0.mlp.experts.down_proj",
        down,
        torch::IntArrayRef({4, 8, 8}),
        /*tp_rank=*/1,
        /*tp_size=*/2);

    auto expected_gate_up =
        torch::cat({gate_up.narrow(1, 8, 8), gate_up.narrow(1, 24, 8)}, /*dim=*/1);
    EXPECT_TRUE(torch::equal(gate_up_shard, expected_gate_up));
    EXPECT_TRUE(torch::equal(down_shard, down.narrow(2, 8, 8)));
}
