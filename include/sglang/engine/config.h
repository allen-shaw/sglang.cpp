#pragma once

#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "sglang/models/config.h"

namespace sglang {

struct EngineConfig {
    std::string model_path;
    torch::Dtype dtype = torch::kFloat16;
    torch::Device device = torch::Device(torch::kCUDA, 0);
    int max_running_req = 256;
    int page_size = 1;
    float memory_ratio = 0.9F;
    bool use_dummy_weight = false;
    bool enable_cuda_graph = false;
    int random_seed = 42;
    std::optional<int> max_seq_len_override;
    std::optional<int> num_pages_override;
    std::vector<int> cuda_graph_batch_sizes;
    std::optional<int> cuda_graph_max_batch_size;
    int cuda_graph_capture_max_seq_len = 4096;
    std::optional<ModelConfig> model_config_override;

    ModelConfig load_model_config() const;
    int max_seq_len() const;
};

}  // namespace sglang
