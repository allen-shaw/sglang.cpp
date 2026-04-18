#pragma once

#include "sglang/attention/backend.h"
#include <torch/torch.h>

namespace sglang {

class BaseKVCachePool;

struct FlashInferAttnMetadata : public BaseAttnMetadata {
    bool is_prefill;

    std::vector<int> seq_lens;
    std::vector<int> cached_lens;

    std::vector<int32_t> qo_indptr_host;
    std::vector<int32_t> kv_indptr_host;
    std::vector<int32_t> indices_host;
    std::vector<int32_t> last_page_len_host;

    torch::Tensor indptr;
    torch::Tensor indices;
    torch::Tensor paged_kv_indptr;
    torch::Tensor paged_kv_last_page_len;

    torch::Tensor get_last_indices(int bs) const override;
};

class FlashInferBackend : public BaseAttnBackend {
public:
    FlashInferBackend(std::shared_ptr<BaseKVCachePool> kv_cache,
                      int num_qo_heads, int num_kv_heads, int head_dim);

    ~FlashInferBackend() override = default;

    void prepare_metadata(Batch& batch) override;

    torch::Tensor forward(const torch::Tensor& q,
                          const torch::Tensor& k,
                          const torch::Tensor& v, int layer_id,
                          Batch& batch) override;

    void init_capture_graph(int max_seq_len,
                            const std::vector<int>& bs_list) override {}

    void prepare_for_capture(Batch& batch) override {}
    void prepare_for_replay(Batch& batch) override {}

private:
    std::shared_ptr<BaseKVCachePool> kv_cache_;
    int num_qo_heads_;
    int num_kv_heads_;
    int head_dim_;

    torch::Tensor float_workspace_;
    torch::Tensor int_workspace_;
    torch::Tensor pinned_int_workspace_;
};

} // namespace sglang
