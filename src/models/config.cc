#include "sglang/models/config.h"

#include <fstream>
#include <nlohmann/json.hpp>

namespace sglang {

ModelConfig ModelConfig::from_json_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    nlohmann::json j;
    ifs >> j;

    ModelConfig config;

    // Required fields
    config.num_layers = j.at("num_hidden_layers").get<int>();
    config.num_qo_heads = j.at("num_attention_heads").get<int>();
    config.num_kv_heads = j.value("num_key_value_heads", config.num_qo_heads);
    config.hidden_size = j.at("hidden_size").get<int>();
    config.vocab_size = j.at("vocab_size").get<int>();
    config.intermediate_size = j.at("intermediate_size").get<int>();
    config.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    config.hidden_act = j.value("hidden_act", std::string("silu"));
    config.tie_word_embeddings = j.value("tie_word_embeddings", false);

    // head_dim: explicit or derived
    if (j.contains("head_dim")) {
        config.head_dim = j["head_dim"].get<int>();
    } else {
        config.head_dim = config.hidden_size / config.num_qo_heads;
    }

    // Rotary config
    config.rotary_config.head_dim = config.head_dim;
    config.rotary_config.rotary_dim = config.head_dim;  // usually same
    config.rotary_config.max_position = j.value("max_position_embeddings", 32768);
    config.rotary_config.base = j.value("rope_theta", 10000.0f);

    // Model type and architectures
    config.model_type = j.value("model_type", std::string(""));
    if (j.contains("architectures")) {
        config.architectures = j["architectures"].get<std::vector<std::string>>();
    }

    // MoE fields (defaults for non-MoE models)
    config.num_experts = j.value("num_experts", 0);
    config.num_experts_per_tok = j.value("num_experts_per_tok", 0);
    config.moe_intermediate_size = j.value("moe_intermediate_size", 0);
    config.norm_topk_prob = j.value("norm_topk_prob", false);

    return config;
}

} // namespace sglang
