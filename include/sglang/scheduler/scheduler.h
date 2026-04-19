#pragma once

#include <memory>
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

class Scheduler {
 public:
    explicit Scheduler(const SchedulerConfig& config);
    ~Scheduler() = default;

    void submit(GenerateRequest request);
    void abort(uint64_t uid);
    std::vector<DetokenizeMsg> step();
    std::vector<DetokenizeMsg> run_until_idle();
    bool has_work() const;

 private:
    ForwardInput prepare_batch(const std::shared_ptr<Batch>& batch);
    std::shared_ptr<Batch> schedule_next_batch();
    ForwardOutput forward(ForwardInput& forward_input, ForwardProfile* profile = nullptr);
    std::vector<DetokenizeMsg> process_forward_output(const ForwardInput& input,
                                                      const ForwardOutput& output,
                                                      ProcessProfile* profile = nullptr);
    void free_req_resources(const std::shared_ptr<Req>& req);

    SchedulerConfig config_;
    Engine engine_;
    TableManager table_manager_;
    CacheManager cache_manager_;
    DecodeManager decode_manager_;
    PrefillManager prefill_manager_;
    int prefill_budget_;
};

}  // namespace sglang
