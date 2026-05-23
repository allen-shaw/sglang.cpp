#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <torch/torch.h>

#include "sglang/models/config.h"

namespace sglang {

/// WeightLoader — loads weights from safetensors files and maps them to model parameters.
/// Handles HuggingFace naming conventions and projection merging (qkv_proj, gate_up_proj).
///
/// Matches Python: minisgl/models/weight.py
class WeightLoader {
 public:
    /// Load weights from a model directory containing *.safetensors files
    /// and apply them to the given model's named parameters.
    ///
    /// @param model The nn::Module to load weights into
    /// @param model_dir Directory containing safetensors files and config.json
    /// @param dtype Target dtype (e.g., torch::kBFloat16)
    /// @param device Target device (e.g., torch::kCUDA)
    static void load_weights(torch::nn::Module& model,
                             const std::string& model_dir,
                             torch::Dtype dtype,
                             torch::Device device,
                             const ModelConfig* model_config = nullptr);

    /// Perform weight merging (q/k/v -> qkv_proj, gate/up -> gate_up_proj)
    /// and MoE expert stacking (experts.N.* -> experts.*).
    /// Returns merged name->tensor map.
    static std::unordered_map<std::string, torch::Tensor>
    merge_weights(std::unordered_map<std::string, torch::Tensor>&& raw_weights);

    static torch::Tensor shard_tensor_for_parameter(const std::string& name,
                                                    const torch::Tensor& tensor,
                                                    const torch::IntArrayRef& param_sizes,
                                                    int tp_rank,
                                                    int tp_size,
                                                    const ModelConfig* model_config = nullptr);

 private:
    /// Read all tensors from a single safetensors file.
    /// Returns a map of tensor_name -> tensor.
    static std::unordered_map<std::string, torch::Tensor>
    read_safetensors(const std::string& filepath, torch::Device device);
};

}  // namespace sglang
