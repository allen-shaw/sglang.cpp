#include "sglang/scheduler/scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

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

std::pair<torch::Tensor, torch::Tensor> make_input_tuple(const Batch& batch,
                                                         const torch::Device& device) {
    auto mapping_host = torch::empty(
        {static_cast<int64_t>(batch.positions.size(0))},
        torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
    auto* mapping_ptr = mapping_host.data_ptr<int64_t>();

    int64_t offset = 0;
    for (const auto& req : batch.padded_reqs) {
        for (int i = 0; i < req->extend_len(); ++i) {
            mapping_ptr[offset++] = req->table_idx;
        }
    }

    return {mapping_host.to(device, /*non_blocking=*/true),
            batch.positions.to(device, torch::kInt64, /*non_blocking=*/true)};
}

std::pair<torch::Tensor, torch::Tensor> make_write_tuple(const Batch& batch,
                                                         const torch::Device& device) {
    auto options = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    auto req_mapping = torch::empty({static_cast<int64_t>(batch.reqs.size())}, options);
    auto write_positions = torch::empty({static_cast<int64_t>(batch.reqs.size())}, options);
    auto* req_mapping_ptr = req_mapping.data_ptr<int64_t>();
    auto* write_positions_ptr = write_positions.data_ptr<int64_t>();
    for (size_t i = 0; i < batch.reqs.size(); ++i) {
        const auto& req = batch.reqs[i];
        req_mapping_ptr[i] = req->table_idx;
        write_positions_ptr[i] = req->can_decode() ? req->device_len() : -1;
    }

    return {req_mapping.to(device, /*non_blocking=*/true),
            write_positions.to(device, /*non_blocking=*/true)};
}

torch::Tensor make_positions(const Batch& batch, const torch::Device& device) {
    const int64_t needed_size = [&]() {
        int64_t total = 0;
        for (const auto& req : batch.padded_reqs) {
            total += req->extend_len();
        }
        return total;
    }();

    auto positions_host = torch::empty(
        {needed_size},
        torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    auto* positions_ptr = positions_host.data_ptr<int32_t>();

    int64_t offset = 0;
    for (const auto& req : batch.padded_reqs) {
        for (int pos = req->cached_len; pos < req->device_len(); ++pos) {
            positions_ptr[offset++] = pos;
        }
    }

    return positions_host.to(device, /*non_blocking=*/true);
}

}  // namespace

Scheduler::Scheduler(const SchedulerConfig& config)
    : config_(config),
      engine_(config),
      table_manager_(config.max_running_req, engine_.page_table()),
      cache_manager_(engine_.num_pages(), config.page_size, engine_.page_table(), config.cache_type),
      decode_manager_(config.page_size),
      prefill_manager_(cache_manager_, table_manager_, decode_manager_),
      prefill_budget_(config.max_extend_tokens) {}

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
    prefill_manager_.add_one_req(std::move(request));
}

void Scheduler::abort(uint64_t uid) {
    auto req_to_free = prefill_manager_.abort_req(uid);
    if (!req_to_free) {
        req_to_free = decode_manager_.abort_req(uid);
    }
    if (req_to_free) {
        free_req_resources(req_to_free);
    }
}

ForwardInput Scheduler::prepare_batch(const std::shared_ptr<Batch>& batch) {
    engine_.pad_batch(*batch);
    cache_manager_.allocate_paged(batch->reqs);
    batch->positions = make_positions(*batch, engine_.device());
    auto input_tuple = make_input_tuple(*batch, engine_.device());
    auto write_tuple = make_write_tuple(*batch, engine_.device());
    batch->out_loc = engine_.page_table().index({input_tuple.first, input_tuple.second});
    batch->input_ids =
        table_manager_.token_pool().index({input_tuple.first, input_tuple.second});
    engine_.prepare_attention_metadata(*batch);
    return ForwardInput{
        batch,
        engine_.prepare_sampling_args(*batch),
        input_tuple,
        write_tuple,
    };
}

std::shared_ptr<Batch> Scheduler::schedule_next_batch() {
    auto batch = prefill_manager_.schedule_next_batch(prefill_budget_);
    if (!batch) {
        batch = decode_manager_.schedule_next_batch();
    }
    return batch;
}

ForwardOutput Scheduler::forward(ForwardInput& forward_input, ForwardProfile* profile) {
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

    decode_manager_.filter_reqs(forward_input.batch->reqs);
    return output;
}

std::vector<DetokenizeMsg> Scheduler::process_forward_output(const ForwardInput& input,
                                                             const ForwardOutput& output,
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
        req->append_host_token(next_token_value);

        bool finished = !req->can_decode();
        if (!req->sampling_params.ignore_eos && engine_.eos_token_id() >= 0) {
            finished = finished || (next_token_value == engine_.eos_token_id());
        }
        reply.push_back(DetokenizeMsg{req->req_id, next_token_value, finished});

        if (finished) {
            decode_manager_.remove_req(req);
            free_req_resources(req);
        } else if (input.batch->is_prefill()) {
            cache_manager_.cache_req(req, /*finished=*/false);
        }
    }

    if (profile != nullptr) {
        const auto host_end = std::chrono::steady_clock::now();
        profile->sync_us = elapsed_us(sync_start, sync_end);
        profile->host_us = elapsed_us(sync_end, host_end);
    }

    return reply;
}

void Scheduler::free_req_resources(const std::shared_ptr<Req>& req) {
    table_manager_.free(req->table_idx);
    cache_manager_.cache_req(req, /*finished=*/true);
}

std::vector<DetokenizeMsg> Scheduler::step() {
    if (!profile_enabled()) {
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
    auto replies = process_forward_output(input, output, &process_profile);
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

std::vector<DetokenizeMsg> Scheduler::run_until_idle() {
    std::vector<DetokenizeMsg> results;
    while (has_work()) {
        auto step_results = step();
        results.insert(results.end(), step_results.begin(), step_results.end());
    }
    return results;
}

bool Scheduler::has_work() const {
    return prefill_manager_.runnable() || decode_manager_.runnable();
}

}  // namespace sglang
