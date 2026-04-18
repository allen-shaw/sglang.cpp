#include "sglang/engine/config.h"

#include <stdexcept>

namespace sglang {

ModelConfig EngineConfig::load_model_config() const {
    if (model_config_override.has_value()) {
        return *model_config_override;
    }
    if (model_path.empty()) {
        throw std::runtime_error(
            "EngineConfig requires either model_path or model_config_override");
    }
    return ModelConfig::from_json_file(model_path + "/config.json");
}

int EngineConfig::max_seq_len() const {
    if (max_seq_len_override.has_value()) {
        return *max_seq_len_override;
    }
    return load_model_config().rotary_config.max_position;
}

}  // namespace sglang
