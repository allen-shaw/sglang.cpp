#pragma once

#include "sglang/messages/messages.h"
#include <torch/torch.h>
#include <memory>

namespace sglang {

// Forward declaration for CacheHandle (to be defined in kvcache)
// Using void* for now as per plan, or a forward declared struct if we assume the name.
// "BaseCacheHandle" is the python name. C++ side might be generic or specific.
// Let's use a forward declared struct.
struct BaseCacheHandle;

struct Req {
    // Unique Request ID
    uint64_t req_id = 0;

    // Input IDs (CPU tensor)
    torch::Tensor input_ids;

    // Index in the request table (if applicable)
    int table_idx = -1;

    // Length of the sequence currently cached
    int cached_len = 0;

    // Expected output length (max new tokens)
    int output_len = 0;

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

    // State update methods
    void complete_one();
    void append_host(const torch::Tensor& next_token);

    // Factory method (optional, or just use constructor)
    // static Req from_proto(const GenerateReq& proto_req, ...);
};

} // namespace sglang
