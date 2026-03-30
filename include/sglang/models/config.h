#pragma once

#include <string>
#include <vector>

namespace sglang {

struct RotaryConfig {
    int head_dim;
    int rotary_dim;
    int max_position;
    float base;
};

struct ModelConfig {
    int num_layers;
    int num_qo_heads;
    int num_kv_heads;
    int head_dim;
    int hidden_size;
    int vocab_size;
    int intermediate_size;
    float rms_norm_eps;
    RotaryConfig rotary_config;
    std::string hidden_act;
    bool tie_word_embeddings;
    int num_experts;
    int num_experts_per_tok;
    int moe_intermediate_size;
    bool norm_topk_prob;
    std::string model_type;
    std::vector<std::string> architectures;

    bool is_moe() const {
        return model_type.find("moe") != std::string::npos;
    }

    /// Load ModelConfig from a HuggingFace config.json file.
    /// Requires nlohmann/json to be available.
    static ModelConfig from_json_file(const std::string& path);
};

} // namespace sglang
