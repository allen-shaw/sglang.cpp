#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>

#include "sglang/attention/backend.h"

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
    torch::Tensor kv_indptr_host_tensor;
    bool initialized = false;
    bool use_capture_buffers = false;
    std::shared_ptr<void> decode_plan_state;

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
                            const std::vector<int>& bs_list) override;

    void prepare_for_capture(Batch& batch) override;
    void prepare_for_replay(Batch& batch) override;

private:
    struct DecodeCaptureData {
        int max_seq_len = 0;
        int max_batch_size = 0;
        std::vector<int> batch_sizes;
        torch::Tensor kv_indptr_host;
        torch::Tensor kv_indptr_device;
        torch::Tensor indices;
        torch::Tensor last_page_len;
        std::unordered_map<int, std::shared_ptr<FlashInferAttnMetadata>> metadata;
    };

    torch::Tensor get_ones_cpu(int bs);
    torch::Tensor get_ones_device(int bs);
    void ensure_decode_workspace(int bs, int64_t total_kv_len);
    void initialize_decode_metadata_once(FlashInferAttnMetadata& metadata, int batch_size);
    std::shared_ptr<FlashInferAttnMetadata> get_capture_metadata(int batch_size) const;

    std::shared_ptr<BaseKVCachePool> kv_cache_;
    int num_qo_heads_;
    int num_kv_heads_;
    int head_dim_;

    torch::Tensor float_workspace_;
    torch::Tensor int_workspace_;
    std::vector<torch::Tensor> pinned_int_workspaces_;
    torch::Tensor cached_ones_cpu_;
    torch::Tensor cached_ones_device_;
    torch::Tensor decode_table_indices_host_;
    torch::Tensor decode_table_indices_device_;
    torch::Tensor decode_kv_indptr_host_;
    torch::Tensor decode_kv_indptr_device_;
    torch::Tensor decode_indices_device_;
    size_t acquire_pinned_int_workspace();
    torch::Tensor& pinned_int_workspace(size_t slot);
    void wait_workspace_copy_done(size_t slot);
    void record_workspace_copy_done(size_t slot);

    size_t next_pinned_int_workspace_slot_ = 0;
    std::vector<std::shared_ptr<at::cuda::CUDAEvent>> workspace_copy_done_events_;
    std::unique_ptr<DecodeCaptureData> decode_capture_;
};

} // namespace sglang
