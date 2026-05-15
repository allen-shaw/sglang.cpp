#include "sglang/models/weight_loader.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <regex>
#include <unordered_map>

#include "sglang/distributed/distributed.h"
#include "safetensors.hpp"

namespace sglang {

namespace {

struct MergeInfo {
    std::string fused_suffix;
    std::vector<std::string> slot_names;
};

const std::unordered_map<std::string, MergeInfo> MERGE_GROUPS = {
    {".q_proj", {".qkv_proj", {"q", "k", "v"}}},
    {".k_proj", {".qkv_proj", {"q", "k", "v"}}},
    {".v_proj", {".qkv_proj", {"q", "k", "v"}}},
    {".gate_proj", {".gate_up_proj", {"gate", "up"}}},
    {".up_proj", {".gate_up_proj", {"gate", "up"}}},
};

const std::unordered_map<std::string, std::string> SLOT_NAMES = {
    {".q_proj", "q"},
    {".k_proj", "k"},
    {".v_proj", "v"},
    {".gate_proj", "gate"},
    {".up_proj", "up"},
};

const std::regex EXPERT_PATTERN(R"(^(.*\.experts)\.(\d+)\.(.+)$)");

bool get_merge_info(const std::string& key,
                    std::string& merged_key,
                    std::string& slot,
                    std::vector<std::string>& all_slots) {
    for (const auto& [suffix, info] : MERGE_GROUPS) {
        auto pos = key.find(suffix);
        if (pos != std::string::npos) {
            merged_key = key.substr(0, pos) + info.fused_suffix + key.substr(pos + suffix.size());
            slot = SLOT_NAMES.at(suffix);
            all_slots = info.slot_names;
            return true;
        }
    }
    return false;
}

std::vector<std::string> find_safetensors_files(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".safetensors") {
            auto filename = entry.path().filename().string();
            if (filename.find("consolidated") == std::string::npos) {
                files.push_back(entry.path().string());
            }
        }
    }
    if (files.empty()) {
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (entry.path().extension() == ".safetensors") {
                files.push_back(entry.path().string());
            }
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::string normalize_weight_name(std::string name) {
    if (name.rfind("language_model.", 0) == 0) {
        name = name.substr(std::string("language_model.").size());
    }
    return name;
}

std::string normalize_param_name(std::string name) {
    if (name.rfind("model.layers_", 0) == 0) {
        auto pos = std::string("model.layers_").size();
        auto end = name.find('.', pos);
        if (end != std::string::npos) {
            name = "model.layers." + name.substr(pos, end - pos) + name.substr(end);
        }
    }
    return name;
}

bool get_expert_stack_info(const std::string& key, std::string& packed_key, int& expert_idx) {
    std::smatch match;
    if (!std::regex_match(key, match, EXPERT_PATTERN)) {
        return false;
    }

    std::string packed_name = match[3].str();
    const std::string weight_suffix = ".weight";
    if (packed_name.size() >= weight_suffix.size() &&
        packed_name.compare(packed_name.size() - weight_suffix.size(), weight_suffix.size(), weight_suffix) == 0) {
        packed_name.resize(packed_name.size() - weight_suffix.size());
    }

    packed_key = match[1].str() + "." + packed_name;
    expert_idx = std::stoi(match[2].str());
    return true;
}

bool is_qkv_weight(const std::string& name) {
    return name.find(".qkv_proj.weight") != std::string::npos;
}

bool is_gate_up_or_col_weight(const std::string& name) {
    return name.find(".gate_up_proj.weight") != std::string::npos;
}

bool is_row_parallel_weight(const std::string& name) {
    return name.find(".down_proj.weight") != std::string::npos ||
           name.find(".o_proj.weight") != std::string::npos;
}

bool is_vocab_parallel_weight(const std::string& name) {
    return name.find("embed_tokens.weight") != std::string::npos ||
           name.find("lm_head.weight") != std::string::npos;
}

torch::Tensor slice_dim(const torch::Tensor& tensor, int64_t dim, int rank, int size) {
    const auto dim_size = tensor.size(dim);
    TORCH_CHECK(dim_size % size == 0, "Cannot shard tensor dimension evenly: ", tensor.sizes());
    const auto local_size = dim_size / size;
    return tensor.narrow(dim, rank * local_size, local_size).contiguous();
}

torch::Tensor shard_qkv_weight(const torch::Tensor& tensor,
                               const torch::IntArrayRef& param_sizes,
                               int rank,
                               int size,
                               const ModelConfig* model_config) {
    TORCH_CHECK((tensor.dim() == 1 || tensor.dim() == 2) && param_sizes.size() == tensor.dim(),
                "QKV weight/bias must be rank-1 or rank-2");
    if (tensor.sizes() == param_sizes) {
        return tensor;
    }
    TORCH_CHECK(model_config != nullptr, "ModelConfig is required to shard qkv_proj weights for TP");
    const int64_t head_dim = model_config->head_dim;
    const int64_t q_rows = static_cast<int64_t>(model_config->num_qo_heads) * head_dim;
    const int64_t kv_rows = static_cast<int64_t>(model_config->num_kv_heads) * head_dim;
    const int64_t local_q_rows = q_rows / size;
    const int64_t local_kv_rows = kv_rows / size;
    TORCH_CHECK(q_rows % size == 0 && kv_rows % size == 0, "QKV heads must be divisible by TP size");
    TORCH_CHECK(tensor.size(0) == q_rows + 2 * kv_rows, "QKV tensor shape does not match config");
    TORCH_CHECK(param_sizes[0] == local_q_rows + 2 * local_kv_rows,
                "QKV local row size mismatch. tensor=", tensor.sizes(), " param=", param_sizes);

    auto q = tensor.narrow(0, rank * local_q_rows, local_q_rows);
    auto k = tensor.narrow(0, q_rows + rank * local_kv_rows, local_kv_rows);
    auto v = tensor.narrow(0, q_rows + kv_rows + rank * local_kv_rows, local_kv_rows);
    return torch::cat({q, k, v}, 0).contiguous();
}

torch::Tensor shard_gate_up_2d(const torch::Tensor& tensor,
                               const torch::IntArrayRef& param_sizes,
                               int rank,
                               int size) {
    TORCH_CHECK(tensor.dim() == 2 && param_sizes.size() == 2, "gate_up weight must be rank-2");
    if (tensor.sizes() == param_sizes) {
        return tensor;
    }
    TORCH_CHECK(tensor.size(0) % 2 == 0, "gate_up first dimension must contain gate and up halves");
    const int64_t half = tensor.size(0) / 2;
    TORCH_CHECK(half % size == 0, "gate/up sizes must be divisible by TP size");
    const int64_t local_half = half / size;
    TORCH_CHECK(param_sizes[0] == local_half * 2,
                "gate_up local row size mismatch. tensor=", tensor.sizes(), " param=", param_sizes);
    auto gate = tensor.narrow(0, rank * local_half, local_half);
    auto up = tensor.narrow(0, half + rank * local_half, local_half);
    return torch::cat({gate, up}, 0).contiguous();
}

torch::Tensor shard_expert_tensor(const std::string& name,
                                  const torch::Tensor& tensor,
                                  const torch::IntArrayRef& param_sizes,
                                  int rank,
                                  int size) {
    TORCH_CHECK(tensor.dim() == 3 && param_sizes.size() == 3, "MoE expert weight must be rank-3");
    if (tensor.sizes() == param_sizes) {
        return tensor;
    }
    if (name.find("gate_up_proj") != std::string::npos) {
        TORCH_CHECK(tensor.size(1) % 2 == 0, "MoE gate_up expert dim must contain gate and up halves");
        const int64_t half = tensor.size(1) / 2;
        TORCH_CHECK(half % size == 0, "MoE gate/up sizes must be divisible by TP size");
        const int64_t local_half = half / size;
        TORCH_CHECK(param_sizes[1] == local_half * 2,
                    "MoE gate_up local shard mismatch. tensor=", tensor.sizes(), " param=", param_sizes);
        auto gate = tensor.narrow(1, rank * local_half, local_half);
        auto up = tensor.narrow(1, half + rank * local_half, local_half);
        return torch::cat({gate, up}, 1).contiguous();
    }
    TORCH_CHECK(tensor.size(2) / size == param_sizes[2],
                "MoE down local shard mismatch. tensor=", tensor.sizes(), " param=", param_sizes);
    return slice_dim(tensor, 2, rank, size);
}

}  // anonymous namespace

std::unordered_map<std::string, torch::Tensor>
WeightLoader::read_safetensors(const std::string& filepath, torch::Device device) {
    return safetensors::load_safetensors(filepath, device);
}

std::unordered_map<std::string, torch::Tensor>
WeightLoader::merge_weights(std::unordered_map<std::string, torch::Tensor>&& raw_weights) {
    std::unordered_map<std::string, torch::Tensor> merged_by_name;
    std::unordered_map<std::string, torch::Tensor> result;
    std::unordered_map<std::string, std::unordered_map<std::string, torch::Tensor>> merge_buf;

    for (auto& [name, tensor] : raw_weights) {
        std::string clean_name = normalize_weight_name(name);
        if (clean_name.rfind("vision_tower.", 0) == 0 || clean_name.rfind("multi_modal_projector.", 0) == 0) {
            continue;
        }

        std::string merged_key, slot;
        std::vector<std::string> all_slots;

        if (get_merge_info(clean_name, merged_key, slot, all_slots)) {
            merge_buf[merged_key][slot] = std::move(tensor);

            bool all_filled = true;
            for (const auto& s : all_slots) {
                if (merge_buf[merged_key].find(s) == merge_buf[merged_key].end()) {
                    all_filled = false;
                    break;
                }
            }
            if (all_filled) {
                std::vector<torch::Tensor> parts;
                for (const auto& s : all_slots) {
                    parts.push_back(merge_buf[merged_key][s]);
                }
                merged_by_name[merged_key] = torch::cat(parts, 0);
                merge_buf.erase(merged_key);
            }
        } else {
            merged_by_name[clean_name] = std::move(tensor);
        }
    }

    if (!merge_buf.empty()) {
        std::string msg = "Incomplete merge groups: ";
        for (const auto& [key, _] : merge_buf) {
            msg += key + " ";
        }
        throw std::runtime_error(msg);
    }

    std::unordered_map<std::string, std::map<int, torch::Tensor>> expert_buf;
    for (auto& [name, tensor] : merged_by_name) {
        std::string packed_key;
        int expert_idx = 0;
        if (get_expert_stack_info(name, packed_key, expert_idx)) {
            expert_buf[packed_key][expert_idx] = std::move(tensor);
        } else {
            result[name] = std::move(tensor);
        }
    }

    for (auto& [packed_key, expert_tensors] : expert_buf) {
        TORCH_CHECK(!expert_tensors.empty(), "Empty expert tensor group for ", packed_key);
        std::vector<torch::Tensor> stacked_parts;
        stacked_parts.reserve(expert_tensors.size());

        int expected_idx = 0;
        for (auto& [expert_idx, tensor] : expert_tensors) {
            if (expert_idx != expected_idx) {
                throw std::runtime_error(
                    "Incomplete expert tensor group for " + packed_key +
                    ": missing expert index " + std::to_string(expected_idx));
            }
            stacked_parts.push_back(std::move(tensor));
            expected_idx++;
        }
        result[packed_key] = torch::stack(stacked_parts, 0);
    }

    return result;
}

torch::Tensor WeightLoader::shard_tensor_for_parameter(const std::string& name,
                                                       const torch::Tensor& tensor,
                                                       const torch::IntArrayRef& param_sizes,
                                                       int rank,
                                                       int size,
                                                       const ModelConfig* model_config) {
    if (tensor.sizes() == param_sizes || size == 1) {
        return tensor;
    }
    if (tensor.dim() == 3) {
        return shard_expert_tensor(name, tensor, param_sizes, rank, size);
    }
    if (is_qkv_weight(name)) {
        return shard_qkv_weight(tensor, param_sizes, rank, size, model_config);
    }
    if (is_gate_up_or_col_weight(name)) {
        return shard_gate_up_2d(tensor, param_sizes, rank, size);
    }
    if (is_vocab_parallel_weight(name) || (tensor.dim() == 1 && tensor.size(0) != param_sizes[0])) {
        return slice_dim(tensor, 0, rank, size);
    }
    if (is_row_parallel_weight(name)) {
        return slice_dim(tensor, 1, rank, size);
    }
    TORCH_CHECK(false,
                "No tensor-parallel shard rule for parameter ", name,
                " tensor=", tensor.sizes(), " param=", param_sizes);
}

void WeightLoader::load_weights(torch::nn::Module& model,
                                const std::string& model_dir,
                                torch::Dtype dtype,
                                torch::Device device,
                                const ModelConfig* model_config) {
    auto files = find_safetensors_files(model_dir);
    if (files.empty()) {
        throw std::runtime_error("No safetensors files found in: " + model_dir);
    }

    model.to(device, dtype);

    std::unordered_map<std::string, torch::Tensor> all_tensors;
    for (const auto& file : files) {
        std::cout << "Loading weights from: " << file << std::endl;
        auto tensors = read_safetensors(file, torch::Device(torch::kCPU));
        for (auto& [name, tensor] : tensors) {
            all_tensors[normalize_weight_name(name)] = std::move(tensor);
        }
    }

    auto merged = merge_weights(std::move(all_tensors));
    std::cout << "Total merged weight tensors: " << merged.size() << std::endl;

    auto params = model.named_parameters();
    std::unordered_map<std::string, torch::Tensor> param_map;
    for (auto& param : params) {
        param_map.emplace(normalize_param_name(param.key()), param.value());
    }

    int loaded = 0;
    int skipped = 0;
    auto tp = get_tp_info();
    for (auto& [name, tensor] : merged) {
        auto it = param_map.find(name);
        if (it == param_map.end()) {
            skipped++;
            continue;
        }

        auto& param = it->second;
        if (param.sizes() != tensor.sizes()) {
            tensor = shard_tensor_for_parameter(name, tensor, param.sizes(), tp.rank, tp.size, model_config);
        }
        if (param.sizes() == tensor.sizes()) {
            param.data().copy_(tensor.to(device, dtype));
            loaded++;
        } else {
            std::cerr << "Shape mismatch for " << name
                      << ": model=" << param.sizes()
                      << " weight=" << tensor.sizes() << std::endl;
            skipped++;
        }
    }

    std::cout << "Loaded " << loaded << " parameters, skipped " << skipped << std::endl;
}

}  // namespace sglang
