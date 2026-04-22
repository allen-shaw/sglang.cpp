#pragma once

#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sglang/engine/engine.h"
#include "sglang/message/message.h"
#include "sglang/message/tokenizer_msg.h"
#include "sglang/scheduler/config.h"
#include "sglang/scheduler/decode.h"
#include "sglang/scheduler/prefill.h"

namespace sglang {

struct ForwardInput {
    std::shared_ptr<Batch> batch;
    BatchSamplingArgs sample_args;
    std::pair<torch::Tensor, torch::Tensor> input_tuple;
    std::pair<torch::Tensor, torch::Tensor> write_tuple;
};

struct ForwardProfile {
    int64_t engine_us = 0;
    int64_t writeback_us = 0;
};

struct ProcessProfile {
    int64_t sync_us = 0;
    int64_t host_us = 0;
};

struct PendingForward {
    ForwardInput input;
    ForwardOutput output;
    ForwardProfile forward_profile;
    std::vector<bool> can_decode_after_forward;
    bool is_decode = false;
    int batch_size = 0;
    int padded_size = 0;
    int64_t schedule_us = 0;
    int64_t prepare_us = 0;
    int64_t forward_us = 0;
};

class Scheduler {
 public:
    explicit Scheduler(const SchedulerConfig& config);
    ~Scheduler();

    void submit(GenerateRequest request);
    void abort(uint64_t uid);
    std::vector<DetokenizeMsg> step();
    std::vector<DetokenizeMsg> run_until_idle();
    bool has_work() const;

 private:
    ForwardInput prepare_batch(const std::shared_ptr<Batch>& batch);
    void ensure_prepare_workspace(int64_t token_count, int64_t req_count);
    std::shared_ptr<Batch> schedule_next_batch();
    ForwardOutput forward(ForwardInput& forward_input, ForwardProfile* profile = nullptr);
    std::vector<DetokenizeMsg> process_forward_output(const ForwardInput& input,
                                                      const ForwardOutput& output,
                                                      const std::vector<bool>* can_decode_after_forward = nullptr,
                                                      ProcessProfile* profile = nullptr);
    std::optional<PendingForward> launch_next_forward();
    std::vector<DetokenizeMsg> step_no_overlap();
    std::vector<DetokenizeMsg> step_overlap();
    bool pending_uses_req(uint64_t uid) const;
    void release_deferred_req(uint64_t uid);
    void flush_deferred_resource_actions();
    void cache_req_or_defer(const std::shared_ptr<Req>& req, bool finished);
    void free_req_resources_or_defer(const std::shared_ptr<Req>& req);
    void free_req_resources(const std::shared_ptr<Req>& req);

    SchedulerConfig config_;
    Engine engine_;
    c10::cuda::CUDAStream scheduler_stream_;
    std::shared_ptr<at::cuda::CUDAEvent> last_prepare_done_event_;
    std::shared_ptr<at::cuda::CUDAEvent> last_forward_done_event_;
    torch::Tensor prepare_positions_i32_host_;
    torch::Tensor prepare_positions_i32_device_;
    torch::Tensor prepare_positions_i64_host_;
    torch::Tensor prepare_positions_i64_device_;
    torch::Tensor prepare_mapping_host_;
    torch::Tensor prepare_mapping_device_;
    torch::Tensor prepare_write_mapping_host_;
    torch::Tensor prepare_write_mapping_device_;
    torch::Tensor prepare_write_positions_host_;
    torch::Tensor prepare_write_positions_device_;
    TableManager table_manager_;
    CacheManager cache_manager_;
    DecodeManager decode_manager_;
    PrefillManager prefill_manager_;
    int prefill_budget_;
    std::optional<PendingForward> pending_forward_;
    std::unordered_set<uint64_t> suppressed_reqs_;
    std::unordered_set<uint64_t> freed_reqs_;
    std::unordered_map<uint64_t, std::shared_ptr<Req>> deferred_free_reqs_;
    std::vector<std::pair<std::shared_ptr<Req>, bool>> deferred_resource_actions_;
};

}  // namespace sglang
