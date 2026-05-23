#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <torch/torch.h>

#include "sglang/core/batch.h"
#include "sglang/kvcache/base.h"
#include "sglang/message/message.h"

namespace sglang {

class BasePrefixCache;
class DecodeManager;

class TableManager {
 public:
    TableManager(int max_running_reqs, const torch::Tensor& page_table);

    int available_size() const;
    int allocate();
    void free(int slot);
    torch::Tensor& token_pool() { return token_pool_; }
    const torch::Tensor& token_pool() const { return token_pool_; }
    const torch::Tensor& page_table() const { return page_table_; }

 private:
    std::vector<int> free_slots_;
    torch::Tensor page_table_;
    torch::Tensor token_pool_;
};

struct PendingReq {
    uint64_t uid = 0;
    torch::Tensor input_ids;
    SamplingParams sampling_params;
    std::shared_ptr<Req> chunked_req;

    int input_len() const;
    int output_len() const;
};

class CacheManager {
 public:
    class LazyFreeRegion {
     public:
        explicit LazyFreeRegion(CacheManager& manager);
        ~LazyFreeRegion();

        LazyFreeRegion(const LazyFreeRegion&) = delete;
        LazyFreeRegion& operator=(const LazyFreeRegion&) = delete;

     private:
        CacheManager& manager_;
    };

    CacheManager(int num_pages, int page_size, const torch::Tensor& page_table,
                 const std::string& cache_type);

    MatchResult match_req(const PendingReq& req);
    int available_size() const;
    void lock(const std::shared_ptr<BaseCacheHandle>& handle);
    void unlock(const std::shared_ptr<BaseCacheHandle>& handle);
    void allocate_paged(const std::vector<std::shared_ptr<Req>>& reqs);
    void cache_req(const std::shared_ptr<Req>& req, bool finished);
    void check_integrity() const;

 private:
    friend class LazyFreeRegion;

    std::vector<int32_t> allocate_pages(int needed_pages);
    void free_indices(const torch::Tensor& indices);
    torch::Tensor pages_to_tokens(const std::vector<int32_t>& pages) const;
    void ensure_page_write_workspace(int needed_tokens);
    void write_page_table(const std::vector<int32_t>& allocated_pages,
                          const std::vector<std::tuple<int, int, int>>& allocation_info);
    void flush_lazy_free();

    int num_pages_;
    int page_size_;
    torch::Tensor page_table_;
    torch::Device device_;
    torch::Tensor page_write_table_idx_host_;
    torch::Tensor page_write_positions_host_;
    torch::Tensor page_write_tokens_host_;
    torch::Tensor page_write_table_idx_device_;
    torch::Tensor page_write_positions_device_;
    torch::Tensor page_write_tokens_device_;
    std::shared_ptr<BasePrefixCache> prefix_cache_;
    std::vector<int32_t> free_slots_;
    bool lazy_free_active_ = false;
    std::vector<torch::Tensor> lazy_free_list_;
};

class PrefillAdder {
 public:
    PrefillAdder(int token_budget, int reserved_size,
                 CacheManager& cache_manager,
                 TableManager& table_manager);

    std::shared_ptr<Req> try_add_one(PendingReq& pending_req);

 private:
    std::optional<std::pair<std::shared_ptr<BaseCacheHandle>, int>>
    try_allocate_one(const PendingReq& req);

    std::shared_ptr<Req> add_one_req(const PendingReq& pending_req,
                                     const std::shared_ptr<BaseCacheHandle>& cache_handle,
                                     int table_idx,
                                     int cached_len);

    int token_budget_;
    int reserved_size_;
    CacheManager& cache_manager_;
    TableManager& table_manager_;
};

class PrefillManager {
 public:
    PrefillManager(CacheManager& cache_manager,
                   TableManager& table_manager,
                   DecodeManager& decode_manager);

    void add_one_req(GenerateRequest req);
    std::shared_ptr<Batch> schedule_next_batch(int prefill_budget);
    std::shared_ptr<Req> abort_req(uint64_t uid);
    bool runnable() const;
    size_t pending_size() const;

 private:
    CacheManager& cache_manager_;
    TableManager& table_manager_;
    DecodeManager& decode_manager_;
    std::vector<PendingReq> pending_list_;
};

}  // namespace sglang
