#include "sglang/scheduler/scheduler.h"

#include <algorithm>

#include <torch/torch.h>

#include "sglang/utils/logger.h"

namespace sglang {

namespace {

std::pair<torch::Tensor, torch::Tensor> make_input_tuple(const Batch& batch,
                                                         const torch::Device& device) {
    auto mapping_host = torch::empty(
        {static_cast<int64_t>(batch.positions.size(0))},
        torch::TensorOptions().dtype(torch::kInt64));
    auto* mapping_ptr = mapping_host.data_ptr<int64_t>();

    int64_t offset = 0;
    for (const auto& req : batch.padded_reqs) {
        for (int i = 0; i < req->extend_len(); ++i) {
            mapping_ptr[offset++] = req->table_idx;
        }
    }

    return {mapping_host.to(device), batch.positions.to(device, torch::kInt64)};
}

std::pair<torch::Tensor, torch::Tensor> make_write_tuple(const Batch& batch,
                                                         const torch::Device& device) {
    std::vector<int64_t> req_mapping;
    std::vector<int64_t> write_positions;
    req_mapping.reserve(batch.reqs.size());
    write_positions.reserve(batch.reqs.size());
    for (const auto& req : batch.reqs) {
        req_mapping.push_back(req->table_idx);
        write_positions.push_back(req->can_decode() ? req->device_len() : -1);
    }

    return {
        torch::tensor(req_mapping, torch::TensorOptions().dtype(torch::kInt64).device(device)),
        torch::tensor(write_positions, torch::TensorOptions().dtype(torch::kInt64).device(device)),
    };
}

torch::Tensor make_positions(const Batch& batch, const torch::Device& device) {
    const int64_t needed_size = [&]() {
        int64_t total = 0;
        for (const auto& req : batch.padded_reqs) {
            total += req->extend_len();
        }
        return total;
    }();

    auto positions_host = torch::empty({needed_size}, torch::TensorOptions().dtype(torch::kInt32));
    auto* positions_ptr = positions_host.data_ptr<int32_t>();

    int64_t offset = 0;
    for (const auto& req : batch.padded_reqs) {
        for (int pos = req->cached_len; pos < req->device_len(); ++pos) {
            positions_ptr[offset++] = pos;
        }
    }

    return positions_host.to(device);
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

ForwardOutput Scheduler::forward(ForwardInput& forward_input) {
    auto& batch = *forward_input.batch;
    batch.input_ids =
        table_manager_.token_pool().index({forward_input.input_tuple.first, forward_input.input_tuple.second});
    auto output = engine_.forward_batch(batch, forward_input.sample_args);

    auto valid_mask = forward_input.write_tuple.second.ge(0);
    if (valid_mask.any().item<bool>()) {
        auto valid_req = forward_input.write_tuple.first.index({valid_mask});
        auto valid_pos = forward_input.write_tuple.second.index({valid_mask});
        auto valid_tokens = output.next_tokens_gpu.index({valid_mask});
        table_manager_.token_pool().index_put_({valid_req, valid_pos}, valid_tokens);
    }

    decode_manager_.filter_reqs(batch.reqs);
    return output;
}

std::vector<DetokenizeMsg> Scheduler::process_forward_output(const ForwardInput& input,
                                                             const ForwardOutput& output) {
    output.synchronize();
    std::vector<DetokenizeMsg> reply;
    CacheManager::LazyFreeRegion lazy_free(cache_manager_);

    for (size_t i = 0; i < input.batch->reqs.size(); ++i) {
        const auto& req = input.batch->reqs[i];
        if (req->is_chunked_prefill) {
            continue;
        }

        auto next_token =
            output.next_tokens_cpu.index({static_cast<int64_t>(i)}).to(torch::kInt32).reshape({1});
        const int32_t next_token_value = next_token.item<int32_t>();
        req->append_host(next_token.cpu());

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

    return reply;
}

void Scheduler::free_req_resources(const std::shared_ptr<Req>& req) {
    table_manager_.free(req->table_idx);
    cache_manager_.cache_req(req, /*finished=*/true);
}

std::vector<DetokenizeMsg> Scheduler::step() {
    auto batch = schedule_next_batch();
    if (!batch) {
        return {};
    }
    auto input = prepare_batch(batch);
    auto output = forward(input);
    return process_forward_output(input, output);
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
