#include "sglang/scheduler/prefill.h"

#include <algorithm>

#include "sglang/kernels/token_pool.h"
#include "sglang/kvcache/cache_manager.h"
#include "sglang/scheduler/decode.h"
#include "sglang/utils/math_utils.h"

namespace sglang {

TableManager::TableManager(int max_running_reqs, const torch::Tensor& page_table)
    : page_table_(page_table),
      token_pool_(torch::zeros_like(page_table, torch::TensorOptions().dtype(torch::kInt32))) {
    free_slots_.reserve(max_running_reqs);
    for (int i = 0; i < max_running_reqs; ++i) {
        free_slots_.push_back(i);
    }
}

int TableManager::available_size() const {
    return static_cast<int>(free_slots_.size());
}

int TableManager::allocate() {
    TORCH_CHECK(!free_slots_.empty(), "TableManager has no free slots");
    int slot = free_slots_.back();
    free_slots_.pop_back();
    return slot;
}

void TableManager::free(int slot) {
    free_slots_.push_back(slot);
}

int PendingReq::input_len() const {
    return static_cast<int>(input_ids.size(0));
}

int PendingReq::output_len() const {
    return sampling_params.max_new_tokens;
}

CacheManager::LazyFreeRegion::LazyFreeRegion(CacheManager& manager)
    : manager_(manager) {
    TORCH_CHECK(!manager_.lazy_free_active_, "Nested lazy free regions are not allowed");
    manager_.lazy_free_active_ = true;
}

CacheManager::LazyFreeRegion::~LazyFreeRegion() {
    manager_.lazy_free_active_ = false;
    manager_.flush_lazy_free();
}

CacheManager::CacheManager(int num_pages, int page_size, const torch::Tensor& page_table,
                           const std::string& cache_type)
    : num_pages_(num_pages),
      page_size_(page_size),
      page_table_(page_table),
      device_(page_table.device()) {
    TORCH_CHECK(cache_type == "radix", "Only radix cache_type is supported");
    prefix_cache_ = std::make_shared<RadixPrefixCache>(page_size);
    free_slots_.reserve(num_pages);
    for (int i = 0; i < num_pages; ++i) {
        free_slots_.push_back(static_cast<int32_t>(i * page_size));
    }
}

MatchResult CacheManager::match_req(const PendingReq& req) {
    TORCH_CHECK(req.input_len() > 0, "Input length must be greater than 0");
    return prefix_cache_->match_prefix(req.input_ids.slice(0, 0, req.input_len() - 1));
}

int CacheManager::available_size() const {
    const auto size_info = prefix_cache_->size_info();
    return static_cast<int>(size_info.evictable_size + free_slots_.size() * page_size_);
}

void CacheManager::lock(const std::shared_ptr<BaseCacheHandle>& handle) {
    prefix_cache_->lock_handle(handle, /*unlock=*/false);
}

void CacheManager::unlock(const std::shared_ptr<BaseCacheHandle>& handle) {
    prefix_cache_->lock_handle(handle, /*unlock=*/true);
}

std::vector<int32_t> CacheManager::allocate_pages(int needed_pages) {
    if (needed_pages <= 0) {
        return {};
    }
    if (needed_pages > static_cast<int>(free_slots_.size())) {
        auto evicted = prefix_cache_->evict((needed_pages - static_cast<int>(free_slots_.size())) * page_size_);
        free_indices(evicted);
    }
    TORCH_CHECK(needed_pages <= static_cast<int>(free_slots_.size()),
                "CacheManager failed to allocate enough pages");

    std::vector<int32_t> allocated(free_slots_.begin(), free_slots_.begin() + needed_pages);
    free_slots_.erase(free_slots_.begin(), free_slots_.begin() + needed_pages);
    return allocated;
}

torch::Tensor CacheManager::pages_to_tokens(const std::vector<int32_t>& pages) const {
    if (pages.empty()) {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    }
    if (page_size_ == 1) {
        return torch::tensor(pages, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    }

    std::vector<int32_t> tokens;
    tokens.reserve(pages.size() * page_size_);
    for (int32_t page : pages) {
        for (int i = 0; i < page_size_; ++i) {
            tokens.push_back(page + i);
        }
    }
    return torch::tensor(tokens, torch::TensorOptions().dtype(torch::kInt32).device(device_));
}

void CacheManager::ensure_page_write_workspace(int needed_tokens) {
    if (page_write_tokens_host_.numel() >= needed_tokens) {
        return;
    }

    int next_len = 1;
    while (next_len < needed_tokens) {
        next_len <<= 1;
    }

    const auto int32_host_options =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    const auto int32_device_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device_);

    page_write_table_idx_host_ = torch::empty({next_len}, int32_host_options);
    page_write_positions_host_ = torch::empty({next_len}, int32_host_options);
    page_write_tokens_host_ = torch::empty({next_len}, int32_host_options);
    page_write_table_idx_device_ = torch::empty({next_len}, int32_device_options);
    page_write_positions_device_ = torch::empty({next_len}, int32_device_options);
    page_write_tokens_device_ = torch::empty({next_len}, int32_device_options);
}

void CacheManager::write_page_table(
    const std::vector<int32_t>& allocated_pages,
    const std::vector<std::tuple<int, int, int>>& allocation_info) {
    if (allocated_pages.empty()) {
        return;
    }

    const int needed_tokens = static_cast<int>(allocated_pages.size()) * page_size_;
    ensure_page_write_workspace(needed_tokens);
    auto table_idx_host = page_write_table_idx_host_.slice(0, 0, needed_tokens);
    auto positions_host = page_write_positions_host_.slice(0, 0, needed_tokens);
    auto tokens_host = page_write_tokens_host_.slice(0, 0, needed_tokens);
    auto table_idx_device = page_write_table_idx_device_.slice(0, 0, needed_tokens);
    auto positions_device = page_write_positions_device_.slice(0, 0, needed_tokens);
    auto tokens_device = page_write_tokens_device_.slice(0, 0, needed_tokens);

    auto* table_idx_ptr = table_idx_host.data_ptr<int32_t>();
    auto* positions_ptr = positions_host.data_ptr<int32_t>();
    auto* tokens_ptr = tokens_host.data_ptr<int32_t>();

    int offset = 0;
    size_t allocated_offset = 0;
    for (const auto& [table_idx, first_page, last_page] : allocation_info) {
        for (int page = first_page; page < last_page; ++page) {
            TORCH_CHECK(allocated_offset < allocated_pages.size(),
                        "Allocated page count mismatch");
            const int32_t page_start = allocated_pages[allocated_offset++];
            for (int i = 0; i < page_size_; ++i) {
                table_idx_ptr[offset] = table_idx;
                positions_ptr[offset] = page * page_size_ + i;
                tokens_ptr[offset] = page_start + i;
                ++offset;
            }
        }
    }
    TORCH_CHECK(offset == needed_tokens, "Allocated token count mismatch");
    TORCH_CHECK(allocated_offset == allocated_pages.size(),
                "Allocated page count mismatch");

    table_idx_device.copy_(table_idx_host, /*non_blocking=*/true);
    positions_device.copy_(positions_host, /*non_blocking=*/true);
    tokens_device.copy_(tokens_host, /*non_blocking=*/true);
    write_token_pool(page_table_, table_idx_device, positions_device, tokens_device);
}

void CacheManager::allocate_paged(const std::vector<std::shared_ptr<Req>>& reqs) {
    int needed_pages = 0;
    std::vector<std::tuple<int, int, int>> allocation_info;
    for (const auto& req : reqs) {
        const int first_page = div_ceil(req->cached_len, page_size_);
        const int last_page = div_ceil(req->device_len(), page_size_);
        if (last_page > first_page) {
            needed_pages += last_page - first_page;
            allocation_info.emplace_back(req->table_idx, first_page, last_page);
        }
    }
    if (needed_pages == 0) {
        return;
    }

    auto allocated = allocate_pages(needed_pages);
    write_page_table(allocated, allocation_info);
}

void CacheManager::free_indices(const torch::Tensor& indices) {
    if (!indices.defined() || indices.numel() == 0) {
        return;
    }
    if (lazy_free_active_) {
        lazy_free_list_.push_back(indices);
        return;
    }

    auto cpu_indices = indices.to(torch::kCPU).contiguous();
    const auto* ptr = cpu_indices.data_ptr<int32_t>();
    for (int64_t i = 0; i < cpu_indices.numel(); i += page_size_) {
        free_slots_.push_back(ptr[i]);
    }
}

void CacheManager::flush_lazy_free() {
    for (const auto& tensor : lazy_free_list_) {
        free_indices(tensor);
    }
    lazy_free_list_.clear();
}

void CacheManager::cache_req(const std::shared_ptr<Req>& req, bool finished) {
    auto insert_ids = req->input_ids.slice(0, 0, req->cached_len);
    auto page_indices = page_table_[req->table_idx].slice(0, 0, req->cached_len);
    auto old_handle = req->cache_handle;
    auto inserted = prefix_cache_->insert_prefix(insert_ids, page_indices);

    if (old_handle) {
        unlock(old_handle);
        if (inserted.cached_len > old_handle->cached_len) {
            free_indices(page_indices.slice(0, old_handle->cached_len, inserted.cached_len));
        }
    }

    if (finished) {
        if (req->cached_len > inserted.handle->cached_len) {
            free_indices(page_indices.slice(0, inserted.handle->cached_len, req->cached_len));
        }
    } else {
        req->cache_handle = inserted.handle;
        lock(inserted.handle);
    }
}

void CacheManager::check_integrity() const {
    prefix_cache_->check_integrity();
    const auto cache_pages = prefix_cache_->size_info().total_size() / page_size_;
    TORCH_CHECK(
        static_cast<int64_t>(free_slots_.size()) + cache_pages == num_pages_,
        "CacheManager integrity check failed");
}

PrefillAdder::PrefillAdder(int token_budget, int reserved_size,
                           CacheManager& cache_manager,
                           TableManager& table_manager)
    : token_budget_(token_budget),
      reserved_size_(reserved_size),
      cache_manager_(cache_manager),
      table_manager_(table_manager) {}

std::optional<std::pair<std::shared_ptr<BaseCacheHandle>, int>>
PrefillAdder::try_allocate_one(const PendingReq& req) {
    if (table_manager_.available_size() == 0) {
        return std::nullopt;
    }

    auto match = cache_manager_.match_req(req);
    auto handle = match.handle;
    const int cached_len = static_cast<int>(handle->cached_len);
    const int extend_len = req.input_len() - cached_len;
    const int estimated_len = extend_len + req.output_len();
    if (estimated_len + reserved_size_ > cache_manager_.available_size()) {
        return std::nullopt;
    }

    cache_manager_.lock(handle);
    if (estimated_len + reserved_size_ > cache_manager_.available_size()) {
        cache_manager_.unlock(handle);
        return std::nullopt;
    }

    const int table_idx = table_manager_.allocate();
    if (cached_len > 0) {
        auto device_ids = table_manager_.token_pool()[table_idx].slice(0, 0, cached_len);
        device_ids.copy_(req.input_ids.slice(0, 0, cached_len).to(device_ids.device()));
        auto page_entry = table_manager_.page_table()[table_idx].slice(0, 0, cached_len);
        page_entry.copy_(handle->get_matched_indices());
    }
    return std::make_pair(handle, table_idx);
}

std::shared_ptr<Req> PrefillAdder::add_one_req(
    const PendingReq& pending_req,
    const std::shared_ptr<BaseCacheHandle>& cache_handle,
    int table_idx,
    int cached_len) {
    const int remain_len = pending_req.input_len() - cached_len;
    const int chunk_size = std::min(token_budget_, remain_len);
    const bool is_chunked = chunk_size < remain_len;
    token_budget_ -= chunk_size;
    reserved_size_ += remain_len + pending_req.output_len();

    const auto slice = pending_req.input_ids.slice(0, cached_len, cached_len + chunk_size);
    auto device_ids =
        table_manager_.token_pool()[table_idx].slice(0, cached_len, cached_len + chunk_size);
    device_ids.copy_(slice.to(device_ids.device()));

    auto req = std::make_shared<Req>();
    req->req_id = pending_req.uid;
    req->input_ids = pending_req.input_ids.slice(0, 0, cached_len + chunk_size).contiguous();
    req->table_idx = table_idx;
    req->cached_len = cached_len;
    req->output_len = pending_req.output_len();
    req->sampling_params = pending_req.sampling_params;
    req->cache_handle = cache_handle;
    req->device_len_ = static_cast<int>(req->input_ids.size(0));
    req->max_device_len_ = req->device_len_ + req->output_len;
    req->is_chunked_prefill = is_chunked;
    req->validate_runtime_state();
    return req;
}

std::shared_ptr<Req> PrefillAdder::try_add_one(PendingReq& pending_req) {
    if (token_budget_ <= 0) {
        return nullptr;
    }

    if (pending_req.chunked_req) {
        return add_one_req(pending_req,
                           pending_req.chunked_req->cache_handle,
                           pending_req.chunked_req->table_idx,
                           pending_req.chunked_req->cached_len);
    }

    auto resource = try_allocate_one(pending_req);
    if (!resource.has_value()) {
        return nullptr;
    }
    return add_one_req(pending_req,
                       resource->first,
                       resource->second,
                       static_cast<int>(resource->first->cached_len));
}

PrefillManager::PrefillManager(CacheManager& cache_manager,
                               TableManager& table_manager,
                               DecodeManager& decode_manager)
    : cache_manager_(cache_manager),
      table_manager_(table_manager),
      decode_manager_(decode_manager) {}

void PrefillManager::add_one_req(GenerateRequest req) {
    pending_list_.push_back(PendingReq{
        req.uid,
        std::move(req.input_ids),
        req.sampling_params,
        nullptr,
    });
}

std::shared_ptr<Batch> PrefillManager::schedule_next_batch(int prefill_budget) {
    if (pending_list_.empty()) {
        return nullptr;
    }

    PrefillAdder adder(
        prefill_budget,
        decode_manager_.inflight_tokens(),
        cache_manager_,
        table_manager_);

    std::vector<std::shared_ptr<Req>> reqs;
    std::vector<PendingReq> chunked_list;
    for (auto& pending_req : pending_list_) {
        auto req = adder.try_add_one(pending_req);
        if (!req) {
            break;
        }
        pending_req.chunked_req.reset();
        if (req->is_chunked_prefill) {
            pending_req.chunked_req = req;
            chunked_list.push_back(pending_req);
        }
        reqs.push_back(req);
    }

    if (reqs.empty()) {
        return nullptr;
    }

    std::vector<PendingReq> new_pending = std::move(chunked_list);
    new_pending.insert(new_pending.end(),
                       pending_list_.begin() + static_cast<long>(reqs.size()),
                       pending_list_.end());
    pending_list_ = std::move(new_pending);

    auto batch = std::make_shared<Batch>();
    batch->reqs = std::move(reqs);
    batch->phase = BatchPhase::Prefill;
    return batch;
}

std::shared_ptr<Req> PrefillManager::abort_req(uint64_t uid) {
    auto it = std::find_if(pending_list_.begin(), pending_list_.end(),
                           [&](const PendingReq& req) { return req.uid == uid; });
    if (it == pending_list_.end()) {
        return nullptr;
    }
    auto chunked_req = it->chunked_req;
    pending_list_.erase(it);
    return chunked_req;
}

bool PrefillManager::runnable() const {
    return !pending_list_.empty();
}

}  // namespace sglang
