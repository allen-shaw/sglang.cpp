#pragma once

#include <memory>
#include <vector>

#include <torch/torch.h>

namespace sglang {

// Forward declaration
struct Batch;

// ============================================================
// Abstract attention metadata
// ============================================================
struct BaseAttnMetadata {
    virtual ~BaseAttnMetadata() = default;
    virtual torch::Tensor get_last_indices(int bs) const = 0;
};

// ============================================================
// Abstract attention backend
// ============================================================
class BaseAttnBackend {
 public:
    virtual ~BaseAttnBackend() = default;

    virtual torch::Tensor forward(const torch::Tensor& q,
                                  const torch::Tensor& k,
                                  const torch::Tensor& v, int layer_id,
                                  Batch& batch) = 0;

    virtual void prepare_metadata(Batch& batch) = 0;

    virtual void init_capture_graph(int max_seq_len,
                                    const std::vector<int>& bs_list) = 0;

    virtual void prepare_for_capture(Batch& batch) = 0;

    virtual void prepare_for_replay(Batch& batch) = 0;
};

}  // namespace sglang
