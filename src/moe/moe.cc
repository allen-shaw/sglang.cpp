#include "sglang/moe/moe.h"

#include "sglang/distributed/distributed.h"

namespace sglang {

MoELayer::MoELayer(int num_experts,
                   int top_k,
                   int hidden_size,
                   int intermediate_size,
                   bool renormalize,
                   std::string activation)
    : num_experts_(num_experts), top_k_(top_k), hidden_size_(hidden_size),
      intermediate_size_(intermediate_size), renormalize_(renormalize),
      activation_(std::move(activation)) {
    TORCH_CHECK(num_experts_ > 0, "MoELayer requires num_experts > 0");
    TORCH_CHECK(top_k_ > 0, "MoELayer requires top_k > 0");
    TORCH_CHECK(top_k_ <= num_experts_, "MoELayer top_k must be <= num_experts");
    TORCH_CHECK(hidden_size_ > 0, "MoELayer requires hidden_size > 0");
    TORCH_CHECK(intermediate_size_ > 0, "MoELayer requires intermediate_size > 0");

    const int local_intermediate_size = divide_even(
        intermediate_size_, tp_size(), "MoELayer intermediate size");

    gate_up_proj = register_parameter(
        "gate_up_proj",
        torch::empty({num_experts_, local_intermediate_size * 2, hidden_size_}));
    down_proj = register_parameter(
        "down_proj",
        torch::empty({num_experts_, hidden_size_, local_intermediate_size}));

    w1 = gate_up_proj;
    w2 = down_proj;
}

torch::Tensor MoELayer::forward(const torch::Tensor& hidden_states, const torch::Tensor& router_logits) {
    TORCH_CHECK(hidden_states.dim() == 2, "MoELayer hidden_states must be [num_tokens, hidden_size]");
    TORCH_CHECK(router_logits.dim() == 2, "MoELayer router_logits must be [num_tokens, num_experts]");
    TORCH_CHECK(hidden_states.size(0) == router_logits.size(0), "MoELayer token count mismatch");
    TORCH_CHECK(hidden_states.size(1) == hidden_size_, "MoELayer hidden size mismatch");
    TORCH_CHECK(router_logits.size(1) == num_experts_, "MoELayer expert count mismatch");
    TORCH_CHECK(gate_up_proj.device() == hidden_states.device(),
                "MoELayer gate_up_proj and hidden_states must be on the same device");
    TORCH_CHECK(down_proj.device() == hidden_states.device(),
                "MoELayer down_proj and hidden_states must be on the same device");

    auto routing_probs = torch::softmax(router_logits.to(torch::kFloat32), -1);
    auto topk = routing_probs.topk(top_k_, -1);
    auto topk_weights = std::get<0>(topk);
    auto topk_ids = std::get<1>(topk).to(torch::kLong);

    if (renormalize_) {
        topk_weights = topk_weights / (topk_weights.sum(-1, true) + 1e-8);
    }

    auto output = torch::zeros_like(hidden_states);
    for (int k = 0; k < top_k_; ++k) {
        auto expert_ids_for_rank = topk_ids.select(1, k);
        auto weights_for_rank = topk_weights.select(1, k).to(hidden_states.scalar_type());

        for (int expert_id = 0; expert_id < num_experts_; ++expert_id) {
            auto token_mask = expert_ids_for_rank.eq(expert_id);
            if (!token_mask.any().item<bool>()) {
                continue;
            }

            auto token_indices = torch::nonzero(token_mask).squeeze(1);
            auto expert_input = hidden_states.index_select(0, token_indices);
            auto gate_up = torch::nn::functional::linear(
                expert_input,
                gate_up_proj.select(0, expert_id),
                torch::Tensor());

            const auto local_intermediate_size = gate_up.size(1) / 2;
            auto gate = gate_up.narrow(1, 0, local_intermediate_size);
            auto up = gate_up.narrow(1, local_intermediate_size, local_intermediate_size);
            torch::Tensor activated;
            if (activation_ == "silu") {
                activated = torch::silu(gate) * up;
            } else if (activation_ == "gelu") {
                activated = torch::gelu(gate) * up;
            } else {
                TORCH_CHECK(false, "Unsupported MoE activation function: ", activation_);
            }

            auto expert_output = torch::nn::functional::linear(
                activated,
                down_proj.select(0, expert_id),
                torch::Tensor());
            expert_output = expert_output * weights_for_rank.index_select(0, token_indices).unsqueeze(1);
            output.index_add_(0, token_indices, expert_output);
        }
    }

    return tensor_model_parallel_all_reduce(output);
}

} // namespace sglang
