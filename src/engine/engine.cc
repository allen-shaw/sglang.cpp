#include "sglang/engine/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include <ATen/ops/cumsum.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime_api.h>
#include <torch/cuda.h>

#include "sglang/attention/flashinfer_backend.h"
#include "sglang/distributed/distributed.h"
#include "sglang/engine/sampling.h"
#include "sglang/kvcache/mha_kvcache.h"
#include "sglang/models/llama.h"
#include "sglang/models/mistral.h"
#include "sglang/models/qwen.h"
#include "sglang/models/qwen3.h"
#include "sglang/models/qwen3_moe.h"
#include "sglang/models/weight_loader.h"
#include "sglang/utils/math_utils.h"

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

struct EngineProfileStats {
    int64_t calls = 0;
    int64_t decode_calls = 0;
    int64_t model_us = 0;
    int64_t select_us = 0;
    int64_t req_state_us = 0;
    int64_t sample_us = 0;
    int64_t copy_event_us = 0;
    int64_t batch_size_sum = 0;

    void add(bool decode,
             int batch_size,
             int64_t model,
             int64_t select,
             int64_t req_state,
             int64_t sample,
             int64_t copy_event) {
        ++calls;
        decode_calls += decode ? 1 : 0;
        batch_size_sum += batch_size;
        model_us += model;
        select_us += select;
        req_state_us += req_state;
        sample_us += sample;
        copy_event_us += copy_event;
        if (calls % 64 == 0) {
            const double denom = static_cast<double>(calls);
            std::cerr << "[sglang.cpp profile] engine calls=" << calls
                      << " decode_calls=" << decode_calls
                      << " avg_model_us=" << model_us / denom
                      << " avg_select_us=" << select_us / denom
                      << " avg_req_state_us=" << req_state_us / denom
                      << " avg_sample_us=" << sample_us / denom
                      << " avg_copy_event_us=" << copy_event_us / denom
                      << " avg_batch_size=" << batch_size_sum / denom
                      << std::endl;
        }
    }
};

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool contains_token(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

std::optional<int> read_optional_env_int(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || std::string(raw).empty()) {
        return std::nullopt;
    }
    return std::stoi(raw);
}

torch::Device resolve_engine_device(const torch::Device& configured_device) {
    if (!configured_device.is_cuda()) {
        return configured_device;
    }
    if (auto local_rank = read_optional_env_int("SGLANG_TP_LOCAL_RANK")) {
        return torch::Device(torch::kCUDA, *local_rank);
    }
    const int world_size =
        read_optional_env_int("SGLANG_TP_SIZE")
            .value_or(read_optional_env_int("WORLD_SIZE").value_or(1));
    if (world_size > 1 && configured_device.index() <= 0) {
        if (auto local_rank = read_optional_env_int("LOCAL_RANK")) {
            return torch::Device(torch::kCUDA, *local_rank);
        }
    }
    return configured_device;
}

torch::Tensor sample_row(torch::Tensor logits,
                         float temperature,
                         int32_t top_k,
                         float top_p) {
    logits = logits / std::max(temperature, 1e-6F);
    if (top_k > 0 && top_k < logits.size(0)) {
        auto topk = torch::topk(logits, top_k);
        auto filtered =
            torch::full_like(logits, -std::numeric_limits<float>::infinity());
        filtered.index_put_({std::get<1>(topk)}, std::get<0>(topk));
        logits = filtered;
    }

    if (top_p < 1.0F) {
        auto sorted = torch::sort(logits, -1, /*descending=*/true);
        auto sorted_logits = std::get<0>(sorted);
        auto sorted_indices = std::get<1>(sorted);
        auto probs = torch::softmax(sorted_logits, -1);
        auto cumulative = torch::cumsum(probs, -1);
        auto cutoff = cumulative > top_p;
        if (cutoff.numel() > 0) {
            cutoff.index_put_({0}, false);
            sorted_logits.masked_fill_(cutoff, -std::numeric_limits<float>::infinity());
            auto filtered =
                torch::full_like(logits, -std::numeric_limits<float>::infinity());
            filtered.index_put_({sorted_indices}, sorted_logits);
            logits = filtered;
        }
    }

    auto probs = torch::softmax(logits, -1);
    return torch::multinomial(probs, 1);
}

torch::Tensor sample_torch_fallback(const torch::Tensor& logits,
                                    const BatchSamplingArgs& args,
                                    int vocab_size) {
    auto logits_fp32 = logits.to(torch::kFloat32);
    std::vector<torch::Tensor> rows;
    rows.reserve(logits_fp32.size(0));
    for (int64_t i = 0; i < logits_fp32.size(0); ++i) {
        const float temperature = args.temperatures[i].item<float>();
        const int32_t top_k =
            args.top_k.defined() ? args.top_k[i].item<int32_t>() : vocab_size;
        const float top_p = args.top_p.defined() ? args.top_p[i].item<float>() : 1.0F;
        rows.push_back(sample_row(logits_fp32[i], temperature, top_k, top_p));
    }
    return torch::cat(rows, 0);
}

torch::Tensor select_sampling_logits(const Batch& batch, const torch::Tensor& logits) {
    if (!batch.is_prefill()) {
        return logits.slice(0, 0, batch.size());
    }
    if (logits.size(0) == batch.size()) {
        return logits;
    }

    std::vector<int64_t> last_indices;
    last_indices.reserve(batch.reqs.size());
    int64_t offset = 0;
    for (size_t i = 0; i < batch.reqs.size(); ++i) {
        const auto& req = batch.reqs[i];
        const int64_t extend_len = req->extend_len();
        TORCH_CHECK(extend_len > 0, "Prefill req must have positive extend_len");
        last_indices.push_back(offset + extend_len - 1);
        offset += extend_len;
    }

    return logits.index_select(
        0,
        torch::tensor(last_indices, torch::TensorOptions().dtype(torch::kInt64).device(logits.device())));
}

}  // namespace

void ForwardOutput::synchronize() const {
    if (copy_done_event) {
        copy_done_event->synchronize();
    }
}

Sampler::Sampler(torch::Device device, int vocab_size)
    : device_(device), vocab_size_(vocab_size) {}

void Sampler::ensure_prepare_workspace(int64_t batch_size) const {
    batch_size = std::max<int64_t>(1, batch_size);
    if (temperatures_host_.numel() >= batch_size) {
        return;
    }

    int64_t next_len = 1;
    while (next_len < batch_size) {
        next_len <<= 1;
    }

    const auto float_host_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
    const auto int32_host_options =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    const auto float_device_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    const auto int32_device_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device_);

    temperatures_host_ = torch::empty({next_len}, float_host_options);
    top_k_host_ = torch::empty({next_len}, int32_host_options);
    top_p_host_ = torch::empty({next_len}, float_host_options);
    for (auto& buffer : temperatures_device_) {
        buffer = torch::empty({next_len}, float_device_options);
    }
    for (auto& buffer : top_k_device_) {
        buffer = torch::empty({next_len}, int32_device_options);
    }
    for (auto& buffer : top_p_device_) {
        buffer = torch::empty({next_len}, float_device_options);
    }
}

BatchSamplingArgs Sampler::prepare(const Batch& batch) const {
    const int64_t batch_size = static_cast<int64_t>(batch.reqs.size());
    ensure_prepare_workspace(batch_size);
    auto temperatures_host = temperatures_host_.slice(0, 0, batch_size);
    auto top_k_host = top_k_host_.slice(0, 0, batch_size);
    auto top_p_host = top_p_host_.slice(0, 0, batch_size);
    const int slot = prepare_slot_;
    prepare_slot_ = (prepare_slot_ + 1) % static_cast<int>(temperatures_device_.size());
    auto temperatures_device = temperatures_device_[slot].slice(0, 0, batch_size);
    auto top_k_device = top_k_device_[slot].slice(0, 0, batch_size);
    auto top_p_device = top_p_device_[slot].slice(0, 0, batch_size);
    auto* temperatures_ptr = temperatures_host.data_ptr<float>();
    auto* top_k_ptr = top_k_host.data_ptr<int32_t>();
    auto* top_p_ptr = top_p_host.data_ptr<float>();

    bool all_greedy = true;
    bool any_top_k = false;
    bool any_top_p = false;
    for (int64_t i = 0; i < batch_size; ++i) {
        const auto& req = batch.reqs[static_cast<size_t>(i)];
        const auto& params = req->sampling_params;
        all_greedy = all_greedy && params.is_greedy();
        temperatures_ptr[i] = std::max(params.is_greedy() ? 0.0F : params.temperature, 1e-6F);
        int32_t top_k = params.top_k >= 1 ? params.top_k : vocab_size_;
        float top_p = std::min(std::max(params.top_p, 1e-6F), 1.0F);
        any_top_k = any_top_k || top_k != vocab_size_;
        any_top_p = any_top_p || top_p < 1.0F;
        top_k_ptr[i] = top_k;
        top_p_ptr[i] = top_p;
    }

    BatchSamplingArgs args;
    if (all_greedy) {
        return args;
    }

    temperatures_device.copy_(temperatures_host, /*non_blocking=*/true);
    args.temperatures = temperatures_device;
    if (any_top_k) {
        top_k_device.copy_(top_k_host, /*non_blocking=*/true);
        args.top_k = top_k_device;
    }
    if (any_top_p) {
        top_p_device.copy_(top_p_host, /*non_blocking=*/true);
        args.top_p = top_p_device;
    }
    return args;
}

torch::Tensor Sampler::sample(const torch::Tensor& logits,
                              const BatchSamplingArgs& args) const {
    if (!args.temperatures.defined()) {
        return torch::argmax(logits, -1);
    }

    if (!logits.is_cuda()) {
        return sample_torch_fallback(logits, args, vocab_size_);
    }

    constexpr int64_t kMaxSamplingRounds = 32;
    const int64_t batch_size = logits.size(0);
    auto logits_fp32 = logits.to(torch::kFloat32);
    auto temperatures = args.temperatures.to(logits.device()).to(torch::kFloat32).view({batch_size, 1});
    auto probs = torch::softmax(logits_fp32 / temperatures, -1).contiguous();
    const bool deterministic = false;

    if (!args.top_k.defined() && !args.top_p.defined()) {
        auto uniform_samples =
            torch::rand({batch_size},
                        torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
        return flashinfer_sample_from_probs(probs, uniform_samples, deterministic);
    }

    auto uniform_samples =
        torch::rand({kMaxSamplingRounds, batch_size},
                    torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
    FlashInferSamplingResult result;
    if (args.top_k.defined() && !args.top_p.defined()) {
        auto top_k = args.top_k.to(logits.device()).to(torch::kFloat32).contiguous();
        result = flashinfer_top_k_sample_from_probs(probs, uniform_samples, top_k, 0, deterministic);
    } else if (!args.top_k.defined() && args.top_p.defined()) {
        auto top_p = args.top_p.to(logits.device()).to(torch::kFloat32).contiguous();
        result = flashinfer_top_p_sample_from_probs(probs, uniform_samples, top_p, 0.0, deterministic);
    } else {
        auto top_k = args.top_k.to(logits.device()).to(torch::kInt32).contiguous();
        auto top_p = args.top_p.to(logits.device()).to(torch::kFloat32).contiguous();
        result = flashinfer_top_k_top_p_sample_from_probs(
            probs, uniform_samples, top_k, 0.0, top_p, 0.0, deterministic);
    }

    if (result.success.defined() && !result.success.all().item<bool>()) {
        return sample_torch_fallback(logits, args, vocab_size_);
    }
    return result.samples;
}

Engine::ModelRunner Engine::create_model_runner(const ModelConfig& model_config) {
    auto choose_name = [&]() {
        if (!model_config.architectures.empty()) {
            return to_lower(model_config.architectures.front());
        }
        return to_lower(model_config.model_type);
    };

    const std::string name = choose_name();
    if (contains_token(name, "qwen3moe") || (model_config.is_moe() && contains_token(name, "qwen3"))) {
        auto model = std::make_shared<Qwen3MoEForCausalLM>(model_config);
        return {model, [model](const torch::Tensor& input_ids, const torch::Tensor& positions) {
                    return model->forward(input_ids, positions);
                }};
    }
    if (contains_token(name, "qwen3")) {
        auto model = std::make_shared<Qwen3ForCausalLM>(model_config);
        return {model, [model](const torch::Tensor& input_ids, const torch::Tensor& positions) {
                    return model->forward(input_ids, positions);
                }};
    }
    if (contains_token(name, "qwen")) {
        auto model = std::make_shared<Qwen2ForCausalLM>(model_config);
        return {model, [model](const torch::Tensor& input_ids, const torch::Tensor& positions) {
                    return model->forward(input_ids, positions);
                }};
    }
    if (contains_token(name, "mistral")) {
        auto model = std::make_shared<MistralForCausalLM>(model_config);
        return {model, [model](const torch::Tensor& input_ids, const torch::Tensor& positions) {
                    return model->forward(input_ids, positions);
                }};
    }

    auto model = std::make_shared<LlamaForCausalLM>(model_config);
    return {model, [model](const torch::Tensor& input_ids, const torch::Tensor& positions) {
                return model->forward(input_ids, positions);
            }};
}

void Engine::initialize_dummy_weights(torch::nn::Module& module,
                                      torch::Device device,
                                      torch::Dtype dtype) {
    torch::NoGradGuard no_grad;
    module.to(device, dtype);
    for (auto& entry : module.named_parameters(/*recurse=*/true)) {
        entry.value().copy_(torch::randn_like(entry.value()));
    }
}

int Engine::determine_num_pages(const EngineConfig& config,
                                const ModelConfig& model_config) {
    if (config.num_pages_override.has_value()) {
        return *config.num_pages_override;
    }

    size_t free_memory = 0;
    size_t total_memory = 0;
    cudaError_t status = cudaMemGetInfo(&free_memory, &total_memory);
    TORCH_CHECK(status == cudaSuccess, "cudaMemGetInfo failed");

    const size_t cache_per_page =
        2ULL * model_config.head_dim *
        div_even(model_config.num_kv_heads, tp_size(), /*allow_replicate=*/false) *
        config.page_size * c10::elementSize(config.dtype) * model_config.num_layers;
    const size_t budget = static_cast<size_t>(free_memory * config.memory_ratio);
    int num_pages = static_cast<int>(budget / std::max<size_t>(cache_per_page, 1ULL));
    num_pages = std::max(num_pages, 2);
    return num_pages;
}

Engine::Engine(const EngineConfig& config)
    : config_(config),
      model_config_(config.load_model_config()),
      device_(resolve_engine_device(config.device)),
      dtype_(config.dtype),
      sampler_(device_, model_config_.vocab_size),
      stream_(c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device_.index())) {
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for Engine");
    TORCH_CHECK(device_.is_cuda(), "Engine only supports CUDA devices");
    TORCH_CHECK(config_.page_size == 1,
                "Current Engine implementation requires page_size == 1");

    c10::cuda::CUDAGuard device_guard(device_);
    torch::manual_seed(config_.random_seed);
    model_config_ = config_.load_model_config();
    model_runner_ = create_model_runner(model_config_);

    if (config_.use_dummy_weight) {
        initialize_dummy_weights(*model_runner_.module, device_, dtype_);
    } else {
        TORCH_CHECK(!config_.model_path.empty(),
                    "model_path is required when use_dummy_weight=false");
        WeightLoader::load_weights(*model_runner_.module, config_.model_path, dtype_, device_, &model_config_);
    }

    num_pages_ = determine_num_pages(config_, model_config_);
    max_seq_len_ = std::min(config_.max_seq_len(), num_pages_ * config_.page_size);
    page_table_ = torch::zeros(
        {config_.max_running_req + 1, align_ceil(max_seq_len_, 32)},
        torch::TensorOptions().dtype(torch::kInt32).device(device_));

    kv_cache_ = std::make_shared<MHAKVCache>(
        divide_even(model_config_.num_kv_heads, tp_size(), "num_kv_heads"),
        model_config_.num_layers,
        model_config_.head_dim,
        num_pages_ + 1,
        config_.page_size,
        dtype_,
        device_);
    attn_backend_ = std::make_shared<FlashInferBackend>(
        kv_cache_,
        divide_even(model_config_.num_qo_heads, tp_size(), "num_qo_heads"),
        divide_even(model_config_.num_kv_heads, tp_size(), "num_kv_heads"),
        model_config_.head_dim);

    ctx_ = std::make_shared<Context>(config_.page_size, attn_backend_);
    ctx_->page_table = page_table_;
    ctx_->kv_cache = kv_cache_;
    reset_global_ctx();
    set_global_ctx(ctx_);

    dummy_req_ = std::make_shared<Req>();
    dummy_req_->req_id = std::numeric_limits<uint64_t>::max();
    dummy_req_->input_ids =
        torch::tensor({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    dummy_req_->table_idx = config_.max_running_req;
    dummy_req_->cached_len = 0;
    dummy_req_->output_len = 1;
    dummy_req_->sampling_params = SamplingParams{};
    dummy_req_->initialize_runtime_state();
    page_table_[dummy_req_->table_idx].fill_(num_pages_ * config_.page_size);

    const int graph_max_bs =
        config_.cuda_graph_max_batch_size.value_or(config_.max_running_req);
    graph_runner_ = std::make_unique<GraphRunner>(
        device_,
        config_.enable_cuda_graph,
        config_.cuda_graph_batch_sizes,
        graph_max_bs,
        page_table_.size(1),
        config_.cuda_graph_capture_max_seq_len,
        model_config_.vocab_size,
        ctx_,
        attn_backend_,
        model_runner_.forward,
        dummy_req_,
        stream_);
}

Engine::~Engine() {
    shutdown();
}

ForwardOutput Engine::forward_batch(Batch& batch, const BatchSamplingArgs& args) {
    c10::InferenceMode inference_guard(true);
    c10::cuda::CUDAStreamGuard stream_guard(stream_);
    std::shared_ptr<Batch> batch_alias(&batch, [](Batch*) {});
    BatchGuard guard(ctx_, batch_alias);

    const bool collect_profile = profile_enabled();
    const auto model_start = std::chrono::steady_clock::now();
    torch::Tensor logits;
    if (graph_runner_->can_use_cuda_graph(batch)) {
        logits = graph_runner_->replay(batch);
    } else {
        auto model_input_ids = batch.input_ids.to(torch::kInt64);
        logits = model_runner_.forward(model_input_ids, batch.positions);
    }
    const auto model_end = std::chrono::steady_clock::now();

    const auto select_start = std::chrono::steady_clock::now();
    auto sampling_logits = select_sampling_logits(batch, logits);
    const auto select_end = std::chrono::steady_clock::now();

    const auto req_state_start = std::chrono::steady_clock::now();
    for (const auto& req : batch.reqs) {
        req->complete_one();
    }
    const auto req_state_end = std::chrono::steady_clock::now();

    const auto sample_start = std::chrono::steady_clock::now();
    auto sampled = sampler_.sample(sampling_logits, args).to(torch::kInt32);
    const auto sample_end = std::chrono::steady_clock::now();
    const auto copy_event_start = std::chrono::steady_clock::now();
    auto next_tokens_cpu = torch::empty(
        sampled.sizes(),
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    next_tokens_cpu.copy_(sampled, /*non_blocking=*/true);
    auto event = std::make_shared<at::cuda::CUDAEvent>();
    event->record(stream_);
    const auto copy_event_end = std::chrono::steady_clock::now();
    if (collect_profile) {
        static thread_local EngineProfileStats profile_stats;
        profile_stats.add(batch.is_decode(),
                          batch.size(),
                          elapsed_us(model_start, model_end),
                          elapsed_us(select_start, select_end),
                          elapsed_us(req_state_start, req_state_end),
                          elapsed_us(sample_start, sample_end),
                          elapsed_us(copy_event_start, copy_event_end));
    }
    return ForwardOutput{sampled, next_tokens_cpu, event};
}

BatchSamplingArgs Engine::prepare_sampling_args(const Batch& batch) const {
    return sampler_.prepare(batch);
}

void Engine::shutdown() {
    if (graph_runner_) {
        graph_runner_->destroy_cuda_graphs();
    }
    reset_global_ctx();
}

void Engine::prepare_attention_metadata(Batch& batch) {
    TORCH_CHECK(attn_backend_, "Engine attention backend is not initialized");
    c10::cuda::CUDAStreamGuard stream_guard(stream_);
    attn_backend_->prepare_metadata(batch);
}

void Engine::pad_batch(Batch& batch) {
    graph_runner_->pad_batch(batch);
}

GraphRunner::GraphRunner(torch::Device device,
                         bool enable_cuda_graph,
                         std::vector<int> batch_sizes,
                         int max_batch_size,
                         int max_seq_len,
                         int capture_max_seq_len,
                         int vocab_size,
                         std::shared_ptr<Context> ctx,
                         std::shared_ptr<BaseAttnBackend> attn_backend,
                         std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> model_forward,
                         std::shared_ptr<Req> dummy_req,
                         c10::cuda::CUDAStream stream)
    : device_(device),
      enable_cuda_graph_(enable_cuda_graph),
      graph_batch_sizes_(determine_batch_sizes(enable_cuda_graph, std::move(batch_sizes), max_batch_size)),
      max_batch_size_(graph_batch_sizes_.empty() ? 0 : graph_batch_sizes_.back()),
      capture_max_seq_len_(capture_max_seq_len > 0 ? std::min(capture_max_seq_len, max_seq_len) : max_seq_len),
      ctx_(std::move(ctx)),
      attn_backend_(std::move(attn_backend)),
      model_forward_(std::move(model_forward)),
      dummy_req_(std::move(dummy_req)),
      stream_(stream) {
    capture_graphs(max_seq_len, vocab_size);
}

std::vector<int> GraphRunner::determine_batch_sizes(bool enable_cuda_graph,
                                                    std::vector<int> batch_sizes,
                                                    int max_batch_size) {
    if (!enable_cuda_graph || max_batch_size < 1) {
        return {};
    }

    if (batch_sizes.empty()) {
        batch_sizes = {1, 2, 4};
        for (int bs = 8; bs <= max_batch_size; bs *= 2) {
            batch_sizes.push_back(bs);
        }
        if (batch_sizes.back() != max_batch_size) {
            batch_sizes.push_back(max_batch_size);
        }
    }

    std::sort(batch_sizes.begin(), batch_sizes.end());
    batch_sizes.erase(
        std::remove_if(batch_sizes.begin(), batch_sizes.end(),
                       [&](int bs) { return bs < 1 || bs > max_batch_size; }),
        batch_sizes.end());
    batch_sizes.erase(std::unique(batch_sizes.begin(), batch_sizes.end()), batch_sizes.end());
    return batch_sizes;
}

bool GraphRunner::can_use_cuda_graph(const Batch& batch) const {
    if (!enable_cuda_graph_ || !batch.is_decode() ||
        graph_batch_sizes_.empty() || batch.size() > max_batch_size_) {
        return false;
    }
    return std::all_of(batch.reqs.begin(), batch.reqs.end(), [this](const auto& req) {
        return req->device_len() <= capture_max_seq_len_;
    });
}

void GraphRunner::pad_batch(Batch& batch) const {
    if (!can_use_cuda_graph(batch)) {
        batch.padded_reqs = batch.reqs;
        return;
    }

    const auto it =
        std::lower_bound(graph_batch_sizes_.begin(), graph_batch_sizes_.end(), batch.size());
    if (it == graph_batch_sizes_.end()) {
        batch.padded_reqs = batch.reqs;
        return;
    }

    batch.padded_reqs = batch.reqs;
    batch.padded_reqs.insert(batch.padded_reqs.end(),
                             static_cast<size_t>(*it - batch.size()),
                             dummy_req_);
}

GraphRunner::GraphCaptureBuffer GraphRunner::GraphCaptureBuffer::init(int max_batch_size,
                                                                      torch::Device device) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    return GraphCaptureBuffer{
        torch::zeros({max_batch_size}, options),
        torch::zeros({max_batch_size}, options),
        torch::zeros({max_batch_size}, options),
    };
}

void GraphRunner::GraphCaptureBuffer::set_batch(Batch& batch, int batch_size) const {
    batch.input_ids = input_ids.slice(0, 0, batch_size);
    batch.out_loc = out_loc.slice(0, 0, batch_size);
    batch.positions = positions.slice(0, 0, batch_size);
}

void GraphRunner::GraphCaptureBuffer::copy_from(const Batch& batch) const {
    const auto count = batch.padded_size();
    input_ids.slice(0, 0, count).copy_(batch.input_ids);
    out_loc.slice(0, 0, count).copy_(batch.out_loc);
    positions.slice(0, 0, count).copy_(batch.positions);
}

void GraphRunner::capture_graphs(int max_seq_len, int vocab_size) {
    graph_map_.clear();
    if (graph_batch_sizes_.empty()) {
        return;
    }

    c10::InferenceMode inference_guard(true);
    std::cerr << "[sglang.cpp graph] capture batch sizes:";
    for (int batch_size : graph_batch_sizes_) {
        std::cerr << " " << batch_size;
    }
    std::cerr << ", capture_max_seq_len=" << capture_max_seq_len_
              << ", model_max_seq_len=" << max_seq_len << std::endl;

    attn_backend_->init_capture_graph(capture_max_seq_len_, graph_batch_sizes_);

    c10::cuda::CUDAGuard device_guard(device_);
    c10::cuda::CUDAStreamGuard stream_guard(stream_);
    torch::cuda::synchronize(device_.index());

    std::optional<at::cuda::MempoolId_t> pool;
    for (auto it = graph_batch_sizes_.rbegin(); it != graph_batch_sizes_.rend(); ++it) {
        const int batch_size = *it;
        std::cerr << "[sglang.cpp graph] capturing bs=" << batch_size << std::endl;
        GraphCaptureState state;
        state.batch_size = batch_size;
        state.batch = std::make_shared<Batch>();
        state.batch->phase = BatchPhase::Decode;
        state.batch->reqs.assign(batch_size, dummy_req_);
        state.batch->padded_reqs = state.batch->reqs;
        state.buffer = GraphCaptureBuffer::init(batch_size, device_);
        state.buffer.set_batch(*state.batch, batch_size);

        attn_backend_->prepare_for_capture(*state.batch);
        state.buffer.out_loc.copy_(
            get_global_ctx()->page_table[dummy_req_->table_idx].slice(0, 0, batch_size));

        {
            BatchGuard guard(ctx_, state.batch);
            state.logits = model_forward_(state.batch->input_ids.to(torch::kInt64),
                                          state.batch->positions);

            state.graph = std::make_unique<at::cuda::CUDAGraph>();
            if (pool.has_value()) {
                state.graph->capture_begin(*pool);
            } else {
                state.graph->capture_begin();
            }
            state.logits = model_forward_(state.batch->input_ids.to(torch::kInt64),
                                          state.batch->positions);
            state.graph->capture_end();
        }

        if (!pool.has_value()) {
            pool = state.graph->pool();
        }
        graph_map_.emplace(batch_size, std::move(state));
        std::cerr << "[sglang.cpp graph] captured bs=" << batch_size << std::endl;
    }
    std::cerr << "[sglang.cpp graph] finished capture" << std::endl;
}

torch::Tensor GraphRunner::replay(Batch& batch) {
    TORCH_CHECK(can_use_cuda_graph(batch), "CUDA graph replay is not available for this batch");
    const auto it = graph_map_.find(batch.padded_size());
    TORCH_CHECK(it != graph_map_.end(), "No CUDA graph captured for batch size ",
                batch.padded_size());

    auto& state = it->second;
    state.buffer.copy_from(batch);
    attn_backend_->prepare_for_replay(batch);
    state.graph->replay();
    return state.logits.slice(0, 0, batch.size());
}

void GraphRunner::destroy_cuda_graphs() {
    graph_map_.clear();
}

}  // namespace sglang
