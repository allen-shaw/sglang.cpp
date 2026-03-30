#include "sglang/models/weight_loader.h"

#include <iostream>
#include <algorithm>
#include <filesystem>
#include <glob.h>

// safetensors.cpp header-only library
#include "safetensors.hpp"

namespace sglang {

namespace {

// Merge group definitions (matching Python weight.py)
struct MergeInfo {
    std::string fused_suffix;
    std::vector<std::string> slot_names;
};

// Map from individual projection suffix -> (fused suffix, all slot names)
const std::unordered_map<std::string, MergeInfo> MERGE_GROUPS = {
    {".q_proj", {".qkv_proj", {"q", "k", "v"}}},
    {".k_proj", {".qkv_proj", {"q", "k", "v"}}},
    {".v_proj", {".qkv_proj", {"q", "k", "v"}}},
    {".gate_proj", {".gate_up_proj", {"gate", "up"}}},
    {".up_proj",   {".gate_up_proj", {"gate", "up"}}},
};

const std::unordered_map<std::string, std::string> SLOT_NAMES = {
    {".q_proj", "q"},
    {".k_proj", "k"},
    {".v_proj", "v"},
    {".gate_proj", "gate"},
    {".up_proj",   "up"},
};

// Check if key contains a merge suffix and return merge info
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

// Find safetensors files in a directory
std::vector<std::string> find_safetensors_files(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".safetensors") {
            auto filename = entry.path().filename().string();
            // Skip consolidated.safetensors
            if (filename.find("consolidated") == std::string::npos) {
                files.push_back(entry.path().string());
            }
        }
    }
    // If no non-consolidated files found, include all safetensors
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

}  // anonymous namespace

std::unordered_map<std::string, torch::Tensor>
WeightLoader::read_safetensors(const std::string& filepath, torch::Device device) {
    return safetensors::load_safetensors(filepath, device);
}

std::unordered_map<std::string, torch::Tensor>
WeightLoader::merge_weights(std::unordered_map<std::string, torch::Tensor>&& raw_weights) {
    std::unordered_map<std::string, torch::Tensor> result;
    
    // Buffer for merge groups: merged_key -> {slot -> tensor}
    std::unordered_map<std::string, std::unordered_map<std::string, torch::Tensor>> merge_buf;

    for (auto& [name, tensor] : raw_weights) {
        // Strip common prefixes
        std::string clean_name = name;
        if (clean_name.find("language_model.") == 0) {
            clean_name = clean_name.substr(std::string("language_model.").size());
        }
        // Skip vision/projector weights
        if (clean_name.find("vision_tower.") == 0 || clean_name.find("multi_modal_projector.") == 0) {
            continue;
        }

        std::string merged_key, slot;
        std::vector<std::string> all_slots;

        if (get_merge_info(clean_name, merged_key, slot, all_slots)) {
            merge_buf[merged_key][slot] = std::move(tensor);

            // Check if all slots are filled
            bool all_filled = true;
            for (const auto& s : all_slots) {
                if (merge_buf[merged_key].find(s) == merge_buf[merged_key].end()) {
                    all_filled = false;
                    break;
                }
            }
            if (all_filled) {
                // Concatenate in order along dim 0
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

    // Load all tensors from all safetensors files
    std::unordered_map<std::string, torch::Tensor> all_tensors;
    for (const auto& file : files) {
        std::cout << "Loading weights from: " << file << std::endl;
        auto tensors = read_safetensors(file, device);
        for (auto& [name, tensor] : tensors) {
            all_tensors[name] = tensor.to(dtype);
        }
    }

    // Merge projections
    auto merged = merge_weights(std::move(all_tensors));
    std::cout << "Total merged weight tensors: " << merged.size() << std::endl;

    // Get model's named parameters
    auto params = model.named_parameters();

    int loaded = 0;
    int skipped = 0;
    for (auto& [name, tensor] : merged) {
        // Try to find matching parameter in model
        bool found = false;
        for (auto& param : params) {
            if (param.key() == name) {
                if (param.value().sizes() == tensor.sizes()) {
                    param.value().data().copy_(tensor);
                    loaded++;
                } else {
                    std::cerr << "Shape mismatch for " << name 
                              << ": model=" << param.value().sizes()
                              << " weight=" << tensor.sizes() << std::endl;
                    skipped++;
                }
                found = true;
                break;
            }
        }
        if (!found) {
            // Not necessarily an error - some weights like rotary cache are buffers
            skipped++;
        }
    }

    std::cout << "Loaded " << loaded << " parameters, skipped " << skipped << std::endl;
}

}  // namespace sglang
