#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "sglang/core/sampling_params.h"
#include "sglang/kvcache/base.h"
#include <torch/torch.h>

namespace sglang {

struct Req {
    // Unique Request ID (also referred to as 'uid' in Python)
    uint64_t req_id = 0;

    // Input IDs (CPU tensor)
    torch::Tensor input_ids;

    // Index in the request table (if applicable)
    int table_idx = -1;

    // Length of the sequence currently cached
    int cached_len = 0;

    // Requested output length (max new tokens)
    int output_len = 0;

    // Runtime lengths tracked independently from host-side input_ids.
    int device_len_ = -1;
    int max_device_len_ = -1;

    // Chunked prefill requests should never enter decode scheduling.
    bool is_chunked_prefill = false;

    // Sampling parameters
    SamplingParams sampling_params;

    // KV Cache handle
    std::shared_ptr<BaseCacheHandle> cache_handle;

    // Calculated fields
    int device_len() const;
    int max_device_len() const;
    int remain_len() const;
    int extend_len() const;
    bool can_decode() const;
    void initialize_runtime_state();
    void validate_runtime_state() const;

    // State update methods
    void complete_one();
    void append_host(const torch::Tensor& next_token);

    // Debug string representation
    std::string toString() const;
};

} // namespace sglang
