#include "sglang/models/weight_loader.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <unordered_map>

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

}  // anonymous namespace

std::unordered_map<std::string, torch::Tensor>
WeightLoader::read_safetensors(const std::string& filepath, torch::Device device) {
    return safetensors::load_safetensors(filepath, device);
}

std::unordered_map<std::string, torch::Tensor>
WeightLoader::merge_weights(std::unordered_map<std::string, torch::Tensor>&& raw_weights) {
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
                result[merged_key] = torch::cat(parts, 0);
                merge_buf.erase(merged_key);
            }
        } else {
            result[clean_name] = std::move(tensor);
        }
    }

    if (!merge_buf.empty()) {
        std::string msg = "Incomplete merge groups: ";
        for (const auto& [key, _] : merge_buf) {
            msg += key + " ";
        }
        throw std::runtime_error(msg);
    }

    return result;
}

void WeightLoader::load_weights(torch::nn::Module& model,
                                const std::string& model_dir,
                                torch::Dtype dtype,
                                torch::Device device) {
    auto files = find_safetensors_files(model_dir);
    if (files.empty()) {
        throw std::runtime_error("No safetensors files found in: " + model_dir);
    }

    model.to(device, dtype);

    std::unordered_map<std::string, torch::Tensor> all_tensors;
    for (const auto& file : files) {
        std::cout << "Loading weights from: " << file << std::endl;
        auto tensors = read_safetensors(file, device);
        for (auto& [name, tensor] : tensors) {
            all_tensors[normalize_weight_name(name)] = tensor.to(dtype);
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
    for (auto& [name, tensor] : merged) {
        auto it = param_map.find(name);
        if (it == param_map.end()) {
            skipped++;
            continue;
        }

        auto& param = it->second;
        if (param.sizes() == tensor.sizes()) {
            param.data().copy_(tensor);
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
