#pragma once

#include "sglang/core/req.h"
#include "sglang/attention/backend.h"
#include <cstdint>
#include <vector>
#include <torch/torch.h>

namespace sglang {

enum class BatchPhase : std::uint8_t {
    Prefill,
    Decode
};

struct Batch {
    std::vector<std::shared_ptr<Req>> reqs;
    BatchPhase phase;

    // These fields should be set by the scheduler
    torch::Tensor input_ids;   // Flattened input IDs
    torch::Tensor positions;   // Position indices
    torch::Tensor out_loc;     // Output locations (slots)
    
    // Subset of reqs that are valid/padded? 
    // Python: padded_reqs: List[Req]
    std::vector<std::shared_ptr<Req>> padded_reqs;

    // Attention metadata (opaque pointer or shared ptr to base class)
    std::shared_ptr<BaseAttnMetadata> attn_metadata;

    // Accessors
    bool is_prefill() const;
    bool is_decode() const;
    int size() const;
    int padded_size() const;
};

} // namespace sglang
