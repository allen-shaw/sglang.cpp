#include "sglang/scheduler/scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

#include <torch/torch.h>

#include <c10/cuda/CUDAGuard.h>

#include "sglang/kernels/token_pool.h"
#include "sglang/utils/logger.h"

namespace sglang {

namespace {

bool profile_enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("SGLANG_CPP_PROFILE");
        return value != nullptr && std::string(value) != "0";
    }();
    return enabled;
}

int64_t elapsed_us(std::chrono::steady_clock::time_point start,
                   std::chrono::steady_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

struct SchedulerProfileStats {
    int64_t steps = 0;
    int64_t decode_steps = 0;
    int64_t total_us = 0;
    int64_t schedule_us = 0;
    int64_t prepare_us = 0;
    int64_t forward_us = 0;
    int64_t engine_us = 0;
    int64_t writeback_us = 0;
    int64_t process_us = 0;
    int64_t process_sync_us = 0;
    int64_t process_host_us = 0;
    int64_t batch_size_sum = 0;
    int64_t padded_size_sum = 0;

    void add(bool decode,
             int batch_size,
             int padded_size,
             int64_t schedule,
             int64_t prepare,
             int64_t forward,
             int64_t engine,
             int64_t writeback,
             int64_t process,
             int64_t process_sync,
             int64_t process_host,
             int64_t total) {
        ++steps;
        decode_steps += decode ? 1 : 0;
        batch_size_sum += batch_size;
        padded_size_sum += padded_size;
        schedule_us += schedule;
        prepare_us += prepare;
        forward_us += forward;
        engine_us += engine;
        writeback_us += writeback;
        process_us += process;
        process_sync_us += process_sync;
        process_host_us += process_host;
        total_us += total;
        if (steps % 64 == 0) {
            const double denom = static_cast<double>(steps);
            std::cerr << "[sglang.cpp profile] scheduler steps=" << steps
                      << " decode_steps=" << decode_steps
                      << " avg_total_us=" << total_us / denom
                      << " avg_schedule_us=" << schedule_us / denom
                      << " avg_prepare_us=" << prepare_us / denom
                      << " avg_forward_us=" << forward_us / denom
                      << " avg_engine_us=" << engine_us / denom
                      << " avg_writeback_us=" << writeback_us / denom
                      << " avg_process_us=" << process_us / denom
                      << " avg_process_sync_us=" << process_sync_us / denom
                      << " avg_process_host_us=" << process_host_us / denom
                      << " avg_batch_size=" << batch_size_sum / denom
                      << " avg_padded_size=" << padded_size_sum / denom
                      << std::endl;
        }
    }
};

struct PrepareProfileStats {
    int64_t steps = 0;
    int64_t pad_alloc_us = 0;
    int64_t workspace_us = 0;
    int64_t h2d_us = 0;
    int64_t gather_us = 0;
    int64_t attn_us = 0;
    int64_t sampling_args_us = 0;

    void add(int64_t pad_alloc,
             int64_t workspace,
             int64_t h2d,
             int64_t gather,
             int64_t attn,
             int64_t sampling_args) {
        ++steps;
        pad_alloc_us += pad_alloc;
        workspace_us += workspace;
        h2d_us += h2d;
        gather_us += gather;
        attn_us += attn;
        sampling_args_us += sampling_args;
        if (steps % 64 == 0) {
            const double denom = static_cast<double>(steps);
            std::cerr << "[sglang.cpp profile] prepare steps=" << steps
                      << " avg_pad_alloc_us=" << pad_alloc_us / denom
                      << " avg_workspace_us=" << workspace_us / denom
                      << " avg_h2d_us=" << h2d_us / denom
                      << " avg_gather_us=" << gather_us / denom
                      << " avg_attn_us=" << attn_us / denom
                      << " avg_sampling_args_us=" << sampling_args_us / denom
                      << std::endl;
        }
    }
};

}  // namespace

Scheduler::Scheduler(const SchedulerConfig& config)
    : config_(config),
      engine_(config),
      scheduler_stream_(c10::cuda::getStreamFromPool(/*isHighPriority=*/false,
                                                     engine_.device().index())),
      table_manager_(config.max_running_req, engine_.page_table()),
      cache_manager_(engine_.num_pages(), config.page_size, engine_.page_table(), config.cache_type),
      decode_manager_(config.page_size),
      prefill_manager_(cache_manager_, table_manager_, decode_manager_),
      prefill_budget_(config.max_extend_tokens) {
    torch::cuda::synchronize(engine_.device().index());
}

Scheduler::~Scheduler() {
    c10::cuda::CUDAGuard device_guard(engine_.device());
    scheduler_stream_.synchronize();
    engine_.stream().synchronize();
}

void Scheduler::submit(GenerateRequest request) {
    TORCH_CHECK(request.input_ids.device().is_cpu(),
                "GenerateRequest.input_ids must be a CPU tensor");
    TORCH_CHECK(request.input_ids.scalar_type() == torch::kInt32,
                "GenerateRequest.input_ids must be int32");
    TORCH_CHECK(request.input_ids.is_contiguous(),
                "GenerateRequest.input_ids must be contiguous");
    TORCH_CHECK(request.input_ids.dim() == 1,
                "GenerateRequest.input_ids must be 1D");

    const int input_len = static_cast<int>(request.input_ids.size(0));
    const int max_output_len = engine_.max_seq_len() - input_len;
    if (max_output_len <= 0 || request.sampling_params.max_new_tokens <= 0) {
        SGLANG_LOG_WARN("Dropping request due to invalid input/output lengths");
        return;
    }
    if (request.sampling_params.max_new_tokens > max_output_len) {
        request.sampling_params.max_new_tokens = max_output_len;
    }
    suppressed_reqs_.erase(request.uid);
    freed_reqs_.erase(request.uid);
    deferred_free_reqs_.erase(request.uid);
    prefill_manager_.add_one_req(std::move(request));
}

void Scheduler::abort(uint64_t uid) {
    auto req_to_free = prefill_manager_.abort_req(uid);
    if (!req_to_free) {
        req_to_free = decode_manager_.abort_req(uid);
    }
    if (req_to_free) {
        suppressed_reqs_.insert(uid);
        if (pending_uses_req(uid)) {
            deferred_free_reqs_[uid] = req_to_free;
        } else {
            free_req_resources(req_to_free);
            suppressed_reqs_.erase(uid);
        }
    }
}

void Scheduler::ensure_prepare_workspace(int64_t token_count, int64_t req_count) {
    token_count = std::max<int64_t>(1, token_count);
    req_count = std::max<int64_t>(1, req_count);
    const auto int32_host_options =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    const auto int64_host_options =
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true);
    const auto int32_device_options =
        torch::TensorOptions().dtype(torch::kInt32).device(engine_.device());
    const auto int64_device_options =
        torch::TensorOptions().dtype(torch::kInt64).device(engine_.device());

    if (prepare_positions_i32_host_.numel() < token_count) {
        int64_t next_len = 1;
        while (next_len < token_count) {
            next_len <<= 1;
        }
        prepare_positions_i32_host_ = torch::empty({next_len}, int32_host_options);
        prepare_positions_i32_device_ = torch::empty({next_len}, int32_device_options);
        prepare_positions_i64_host_ = torch::empty({next_len}, int64_host_options);
        prepare_positions_i64_device_ = torch::empty({next_len}, int64_device_options);
        prepare_mapping_host_ = torch::empty({next_len}, int64_host_options);
        prepare_mapping_device_ = torch::empty({next_len}, int64_device_options);
    }

    if (prepare_write_mapping_host_.numel() < req_count) {
        int64_t next_len = 1;
        while (next_len < req_count) {
            next_len <<= 1;
        }
        prepare_write_mapping_host_ = torch::empty({next_len}, int64_host_options);
        prepare_write_mapping_device_ = torch::empty({next_len}, int64_device_options);
        prepare_write_positions_host_ = torch::empty({next_len}, int64_host_options);
        prepare_write_positions_device_ = torch::empty({next_len}, int64_device_options);
    }
}

ForwardInput Scheduler::prepare_batch(const std::shared_ptr<Batch>& batch) {
    c10::cuda::CUDAStreamGuard stream_guard(scheduler_stream_);
    const bool collect_profile = profile_enabled();
    const auto pad_alloc_start = std::chrono::steady_clock::now();
    engine_.pad_batch(*batch);
    cache_manager_.allocate_paged(batch->reqs);
    const auto pad_alloc_end = std::chrono::steady_clock::now();

    const auto workspace_start = std::chrono::steady_clock::now();
    int64_t token_count = 0;
    for (const auto& req : batch->padded_reqs) {
        token_count += req->extend_len();
    }
    ensure_prepare_workspace(token_count, static_cast<int64_t>(batch->reqs.size()));

    auto positions_i32_host = prepare_positions_i32_host_.slice(0, 0, token_count);
    auto positions_i32_device = prepare_positions_i32_device_.slice(0, 0, token_count);
    auto positions_i64_host = prepare_positions_i64_host_.slice(0, 0, token_count);
    auto positions_i64_device = prepare_positions_i64_device_.slice(0, 0, token_count);
    auto mapping_host = prepare_mapping_host_.slice(0, 0, token_count);
    auto mapping_device = prepare_mapping_device_.slice(0, 0, token_count);

    auto* positions_i32_ptr = positions_i32_host.data_ptr<int32_t>();
    auto* positions_i64_ptr = positions_i64_host.data_ptr<int64_t>();
    auto* mapping_ptr = mapping_host.data_ptr<int64_t>();
    int64_t offset = 0;
    for (const auto& req : batch->padded_reqs) {
        for (int pos = req->cached_len; pos < req->device_len(); ++pos) {
            positions_i32_ptr[offset] = pos;
            positions_i64_ptr[offset] = pos;
            mapping_ptr[offset] = req->table_idx;
            ++offset;
        }
    }
    TORCH_CHECK(offset == token_count, "Prepared token count mismatch");
    const auto workspace_end = std::chrono::steady_clock::now();

    const auto h2d_start = std::chrono::steady_clock::now();
    positions_i32_device.copy_(positions_i32_host, /*non_blocking=*/true);
    positions_i64_device.copy_(positions_i64_host, /*non_blocking=*/true);
    mapping_device.copy_(mapping_host, /*non_blocking=*/true);
    batch->positions = positions_i32_device;
    auto input_tuple = std::make_pair(mapping_device, positions_i64_device);

    const int64_t req_count = static_cast<int64_t>(batch->reqs.size());
    auto write_mapping_host = prepare_write_mapping_host_.slice(0, 0, req_count);
    auto write_mapping_device = prepare_write_mapping_device_.slice(0, 0, req_count);
    auto write_positions_host = prepare_write_positions_host_.slice(0, 0, req_count);
    auto write_positions_device = prepare_write_positions_device_.slice(0, 0, req_count);
    auto* write_mapping_ptr = write_mapping_host.data_ptr<int64_t>();
    auto* write_positions_ptr = write_positions_host.data_ptr<int64_t>();
    for (int64_t i = 0; i < req_count; ++i) {
        const auto& req = batch->reqs[static_cast<size_t>(i)];
        write_mapping_ptr[i] = req->table_idx;
        write_positions_ptr[i] = req->can_decode() ? req->device_len() : -1;
    }
    write_mapping_device.copy_(write_mapping_host, /*non_blocking=*/true);
    write_positions_device.copy_(write_positions_host, /*non_blocking=*/true);
    auto write_tuple = std::make_pair(write_mapping_device, write_positions_device);
    const auto h2d_end = std::chrono::steady_clock::now();

    const auto gather_start = std::chrono::steady_clock::now();
    batch->out_loc = gather_int32_2d(engine_.page_table(), input_tuple.first, input_tuple.second);
    batch->input_ids =
        gather_int32_2d(table_manager_.token_pool(), input_tuple.first, input_tuple.second);
    const auto gather_end = std::chrono::steady_clock::now();

    const auto attn_start = std::chrono::steady_clock::now();
    engine_.prepare_attention_metadata(*batch);
    const auto attn_end = std::chrono::steady_clock::now();
    const auto sampling_args_start = std::chrono::steady_clock::now();
    auto sample_args = engine_.prepare_sampling_args(*batch);
    const auto sampling_args_end = std::chrono::steady_clock::now();
    if (collect_profile) {
        static thread_local PrepareProfileStats profile_stats;
        profile_stats.add(elapsed_us(pad_alloc_start, pad_alloc_end),
                          elapsed_us(workspace_start, workspace_end),
                          elapsed_us(h2d_start, h2d_end),
                          elapsed_us(gather_start, gather_end),
                          elapsed_us(attn_start, attn_end),
                          elapsed_us(sampling_args_start, sampling_args_end));
    }
    return ForwardInput{
        batch,
        std::move(sample_args),
        input_tuple,
        write_tuple,
    };
}

std::shared_ptr<Batch> Scheduler::schedule_next_batch() {
    c10::cuda::CUDAStreamGuard stream_guard(scheduler_stream_);
    if (last_forward_done_event_) {
        last_forward_done_event_->block(scheduler_stream_);
    }

    auto batch = prefill_manager_.schedule_next_batch(prefill_budget_);
    if (!batch) {
        batch = decode_manager_.schedule_next_batch();
    }
    return batch;
}

ForwardOutput Scheduler::forward(ForwardInput& forward_input, ForwardProfile* profile) {
    auto prepare_done_event = std::make_shared<at::cuda::CUDAEvent>();
    prepare_done_event->record(scheduler_stream_);
    prepare_done_event->block(engine_.stream());
    last_prepare_done_event_ = prepare_done_event;

    const auto engine_start = std::chrono::steady_clock::now();
    auto output = engine_.forward_batch(*forward_input.batch, forward_input.sample_args);
    const auto engine_end = std::chrono::steady_clock::now();

    const auto writeback_start = std::chrono::steady_clock::now();
    {
        c10::cuda::CUDAStreamGuard stream_guard(engine_.stream());
        write_token_pool(table_manager_.token_pool(),
                         forward_input.write_tuple.first,
                         forward_input.write_tuple.second,
                         output.next_tokens_gpu);
        if (output.copy_done_event) {
            output.copy_done_event->record(engine_.stream());
        }
    }
    const auto writeback_end = std::chrono::steady_clock::now();
    if (profile != nullptr) {
        profile->engine_us = elapsed_us(engine_start, engine_end);
        profile->writeback_us = elapsed_us(writeback_start, writeback_end);
    }

    last_forward_done_event_ = output.copy_done_event;
    decode_manager_.filter_reqs(forward_input.batch->reqs);
    return output;
}

std::vector<DetokenizeMsg> Scheduler::process_forward_output(const ForwardInput& input,
                                                             const ForwardOutput& output,
                                                             const std::vector<bool>* can_decode_after_forward,
                                                             ProcessProfile* profile) {
    const auto sync_start = std::chrono::steady_clock::now();
    output.synchronize();
    const auto sync_end = std::chrono::steady_clock::now();
    std::vector<DetokenizeMsg> reply;
    CacheManager::LazyFreeRegion lazy_free(cache_manager_);

    auto* next_tokens = output.next_tokens_cpu.data_ptr<int32_t>();
    for (size_t i = 0; i < input.batch->reqs.size(); ++i) {
        const auto& req = input.batch->reqs[i];
        if (req->is_chunked_prefill) {
            continue;
        }

        const int32_t next_token_value = next_tokens[i];
        const bool suppressed = suppressed_reqs_.count(req->req_id) > 0;
        if (suppressed) {
            release_deferred_req(req->req_id);
            continue;
        }

        req->append_host_token(next_token_value);

        const bool can_decode =
            can_decode_after_forward != nullptr ? (*can_decode_after_forward)[i] : req->can_decode();
        bool finished = !can_decode;
        if (!req->sampling_params.ignore_eos && engine_.eos_token_id() >= 0) {
            finished = finished || (next_token_value == engine_.eos_token_id());
        }

        if (finished) {
            decode_manager_.remove_req(req);
            if (pending_uses_req(req->req_id)) {
                suppressed_reqs_.insert(req->req_id);
                deferred_free_reqs_[req->req_id] = req;
            } else {
                free_req_resources_or_defer(req);
            }
        } else if (input.batch->is_prefill() && !pending_uses_req(req->req_id)) {
            cache_req_or_defer(req, /*finished=*/false);
        }

        reply.push_back(DetokenizeMsg{req->req_id, next_token_value, finished});
    }

    if (profile != nullptr) {
        const auto host_end = std::chrono::steady_clock::now();
        profile->sync_us = elapsed_us(sync_start, sync_end);
        profile->host_us = elapsed_us(sync_end, host_end);
    }

    return reply;
}

std::optional<PendingForward> Scheduler::launch_next_forward() {
    flush_deferred_resource_actions();

    const auto schedule_start = std::chrono::steady_clock::now();
    auto batch = schedule_next_batch();
    const auto schedule_end = std::chrono::steady_clock::now();
    if (!batch) {
        return std::nullopt;
    }

    PendingForward pending;
    pending.is_decode = batch->is_decode();

    const auto prepare_start = std::chrono::steady_clock::now();
    pending.input = prepare_batch(batch);
    const auto prepare_end = std::chrono::steady_clock::now();

    const auto forward_start = std::chrono::steady_clock::now();
    pending.output = forward(pending.input, &pending.forward_profile);
    const auto forward_end = std::chrono::steady_clock::now();

    pending.can_decode_after_forward.reserve(pending.input.batch->reqs.size());
    for (const auto& req : pending.input.batch->reqs) {
        pending.can_decode_after_forward.push_back(req->can_decode());
    }
    pending.batch_size = static_cast<int>(pending.input.batch->reqs.size());
    pending.padded_size = static_cast<int>(pending.input.batch->padded_reqs.size());
    pending.schedule_us = elapsed_us(schedule_start, schedule_end);
    pending.prepare_us = elapsed_us(prepare_start, prepare_end);
    pending.forward_us = elapsed_us(forward_start, forward_end);
    return pending;
}

bool Scheduler::pending_uses_req(uint64_t uid) const {
    if (!pending_forward_.has_value()) {
        return false;
    }
    for (const auto& req : pending_forward_->input.batch->reqs) {
        if (req->req_id == uid) {
            return true;
        }
    }
    return false;
}

void Scheduler::release_deferred_req(uint64_t uid) {
    auto it = deferred_free_reqs_.find(uid);
    if (it != deferred_free_reqs_.end()) {
        free_req_resources_or_defer(it->second);
        deferred_free_reqs_.erase(it);
    }
    suppressed_reqs_.erase(uid);
}

void Scheduler::flush_deferred_resource_actions() {
    if (deferred_resource_actions_.empty()) {
        return;
    }
    if (last_forward_done_event_) {
        last_forward_done_event_->synchronize();
    }

    auto actions = std::move(deferred_resource_actions_);
    deferred_resource_actions_.clear();
    for (const auto& [req, finished] : actions) {
        if (finished) {
            free_req_resources(req);
        } else {
            cache_manager_.cache_req(req, /*finished=*/false);
        }
    }
}

void Scheduler::cache_req_or_defer(const std::shared_ptr<Req>& req, bool finished) {
    if (config_.enable_overlap_scheduling && pending_forward_.has_value()) {
        deferred_resource_actions_.push_back({req, finished});
        return;
    }
    cache_manager_.cache_req(req, finished);
}

void Scheduler::free_req_resources_or_defer(const std::shared_ptr<Req>& req) {
    if (config_.enable_overlap_scheduling && pending_forward_.has_value()) {
        deferred_resource_actions_.push_back({req, true});
        return;
    }
    free_req_resources(req);
}

void Scheduler::free_req_resources(const std::shared_ptr<Req>& req) {
    if (!freed_reqs_.insert(req->req_id).second) {
        return;
    }
    table_manager_.free(req->table_idx);
    cache_manager_.cache_req(req, /*finished=*/true);
}

std::vector<DetokenizeMsg> Scheduler::step_no_overlap() {
    const bool collect_profile = profile_enabled();
    if (!collect_profile) {
        auto batch = schedule_next_batch();
        if (!batch) {
            return {};
        }
        auto input = prepare_batch(batch);
        auto output = forward(input);
        return process_forward_output(input, output);
    }

    static thread_local SchedulerProfileStats profile_stats;
    const auto total_start = std::chrono::steady_clock::now();
    const auto schedule_start = std::chrono::steady_clock::now();
    auto batch = schedule_next_batch();
    const auto schedule_end = std::chrono::steady_clock::now();
    if (!batch) {
        return {};
    }
    const bool is_decode = batch->is_decode();

    const auto prepare_start = std::chrono::steady_clock::now();
    auto input = prepare_batch(batch);
    const auto prepare_end = std::chrono::steady_clock::now();

    ForwardProfile forward_profile;
    const auto forward_start = std::chrono::steady_clock::now();
    auto output = forward(input, &forward_profile);
    const auto forward_end = std::chrono::steady_clock::now();

    const auto process_start = std::chrono::steady_clock::now();
    ProcessProfile process_profile;
    auto replies = process_forward_output(input, output, nullptr, &process_profile);
    const auto process_end = std::chrono::steady_clock::now();
    const auto total_end = process_end;

    profile_stats.add(
        is_decode,
        static_cast<int>(input.batch->reqs.size()),
        static_cast<int>(input.batch->padded_reqs.size()),
        elapsed_us(schedule_start, schedule_end),
        elapsed_us(prepare_start, prepare_end),
        elapsed_us(forward_start, forward_end),
        forward_profile.engine_us,
        forward_profile.writeback_us,
        elapsed_us(process_start, process_end),
        process_profile.sync_us,
        process_profile.host_us,
        elapsed_us(total_start, total_end));
    return replies;
}

std::vector<DetokenizeMsg> Scheduler::step_overlap() {
    const bool collect_profile = profile_enabled();
    const auto total_start = std::chrono::steady_clock::now();

    auto previous = std::move(pending_forward_);
    pending_forward_.reset();
    pending_forward_ = launch_next_forward();

    if (!previous.has_value()) {
        return {};
    }

    const auto process_start = std::chrono::steady_clock::now();
    ProcessProfile process_profile;
    auto replies = process_forward_output(previous->input,
                                          previous->output,
                                          &previous->can_decode_after_forward,
                                          collect_profile ? &process_profile : nullptr);
    const auto process_end = std::chrono::steady_clock::now();

    if (collect_profile) {
        static thread_local SchedulerProfileStats profile_stats;
        profile_stats.add(
            previous->is_decode,
            previous->batch_size,
            previous->padded_size,
            previous->schedule_us,
            previous->prepare_us,
            previous->forward_us,
            previous->forward_profile.engine_us,
            previous->forward_profile.writeback_us,
            elapsed_us(process_start, process_end),
            process_profile.sync_us,
            process_profile.host_us,
            elapsed_us(total_start, process_end));
    }

    return replies;
}

std::vector<DetokenizeMsg> Scheduler::step() {
    if (config_.enable_overlap_scheduling) {
        return step_overlap();
    }
    return step_no_overlap();
}

std::vector<DetokenizeMsg> Scheduler::run_until_idle() {
    std::vector<DetokenizeMsg> results;
    while (has_work()) {
        auto step_results = step();
        results.insert(results.end(), step_results.begin(), step_results.end());
    }
    return results;
}

bool Scheduler::has_work() const {
    return pending_forward_.has_value() || !deferred_resource_actions_.empty() ||
           prefill_manager_.runnable() || decode_manager_.runnable();
}

}  // namespace sglang
