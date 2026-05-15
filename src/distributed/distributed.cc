#include "sglang/distributed/distributed.h"

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

namespace sglang {
namespace {

std::mutex g_tp_mutex;
std::optional<TensorParallelInfo> g_test_tp_info;

struct ProcessGroupState {
    TensorParallelInfo info;
    std::string master_addr;
    int master_port = 29500;
    c10::intrusive_ptr<c10d::Store> store;
    c10::intrusive_ptr<c10d::Backend> process_group;
};

std::mutex g_process_group_mutex;
std::optional<ProcessGroupState> g_process_group_state;

int read_env_int(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (!raw || std::string(raw).empty()) {
        return default_value;
    }
    return std::stoi(raw);
}

std::string read_env_string(const char* name, const std::string& default_value) {
    const char* raw = std::getenv(name);
    if (!raw || std::string(raw).empty()) {
        return default_value;
    }
    return raw;
}

TensorParallelInfo read_tp_info_from_env() {
    TensorParallelInfo info;
    info.size = read_env_int("SGLANG_TP_SIZE", read_env_int("WORLD_SIZE", 1));
    info.rank = read_env_int("SGLANG_TP_RANK", read_env_int("RANK", 0));
    TORCH_CHECK(info.size >= 1, "Tensor parallel size must be >= 1");
    TORCH_CHECK(info.rank >= 0 && info.rank < info.size,
                "Tensor parallel rank must be in [0, size)");
    return info;
}

std::string master_addr_from_env() {
    return read_env_string("SGLANG_TP_MASTER_ADDR", read_env_string("MASTER_ADDR", "127.0.0.1"));
}

int master_port_from_env() {
    int port = read_env_int("SGLANG_TP_MASTER_PORT", read_env_int("MASTER_PORT", 29500));
    TORCH_CHECK(port > 0 && port <= 65535, "Invalid tensor-parallel master port: ", port);
    return port;
}

ProcessGroupState create_process_group(const TensorParallelInfo& info) {
#ifndef USE_C10D_NCCL
    TORCH_CHECK(false, "This LibTorch build does not include c10d NCCL support");
#else
    const std::string master_addr = master_addr_from_env();
    const int master_port = master_port_from_env();
    const int timeout_sec = read_env_int("SGLANG_TP_TIMEOUT_SEC", 600);
    TORCH_CHECK(timeout_sec > 0, "SGLANG_TP_TIMEOUT_SEC must be positive");

    c10d::TCPStoreOptions store_options;
    store_options.port = static_cast<std::uint16_t>(master_port);
    store_options.isServer = info.rank == 0;
    store_options.numWorkers = static_cast<std::size_t>(info.size);
    store_options.waitWorkers = true;
    store_options.timeout = std::chrono::seconds(timeout_sec);
    store_options.multiTenant = true;

    auto store = c10::make_intrusive<c10d::TCPStore>(master_addr, store_options);
    auto options = c10d::ProcessGroupNCCL::Options::create();
    options->timeout = std::chrono::seconds(timeout_sec);
    auto process_group = c10::make_intrusive<c10d::ProcessGroupNCCL>(
        store,
        info.rank,
        info.size,
        options);

    return ProcessGroupState{info, master_addr, master_port, store, process_group};
#endif
}

const ProcessGroupState& get_or_create_process_group(const TensorParallelInfo& info) {
    std::lock_guard<std::mutex> lock(g_process_group_mutex);
    const std::string master_addr = master_addr_from_env();
    const int master_port = master_port_from_env();
    if (!g_process_group_state.has_value() ||
        g_process_group_state->info.rank != info.rank ||
        g_process_group_state->info.size != info.size ||
        g_process_group_state->master_addr != master_addr ||
        g_process_group_state->master_port != master_port) {
        g_process_group_state = create_process_group(info);
    }
    return *g_process_group_state;
}

}  // namespace

TensorParallelInfo get_tp_info() {
    std::lock_guard<std::mutex> lock(g_tp_mutex);
    if (g_test_tp_info.has_value()) {
        return *g_test_tp_info;
    }
    return read_tp_info_from_env();
}

int tp_rank() {
    return get_tp_info().rank;
}

int tp_size() {
    return get_tp_info().size;
}

void set_tp_info_for_test(int rank, int size) {
    TORCH_CHECK(size >= 1, "Tensor parallel size must be >= 1");
    TORCH_CHECK(rank >= 0 && rank < size, "Tensor parallel rank must be in [0, size)");
    std::lock_guard<std::mutex> lock(g_tp_mutex);
    g_test_tp_info = TensorParallelInfo{rank, size};
}

void reset_tp_info_for_test() {
    std::lock_guard<std::mutex> lock(g_tp_mutex);
    g_test_tp_info.reset();
}

torch::Tensor tensor_model_parallel_all_reduce(const torch::Tensor& x) {
    auto info = get_tp_info();
    if (!info.enabled()) {
        return x;
    }
    TORCH_CHECK(x.is_cuda(), "Tensor-parallel all-reduce requires a CUDA tensor");

    c10::cuda::OptionalCUDAGuard device_guard(x.device());
    auto out = x.contiguous();
    std::vector<at::Tensor> tensors{out};

    c10d::AllreduceOptions options;
    options.reduceOp = c10d::ReduceOp::SUM;
    options.asyncOp = false;
    auto work = get_or_create_process_group(info).process_group->allreduce(tensors, options);
    work->wait();
    return out;
}

int divide_even(int value, int divisor, const char* name) {
    TORCH_CHECK(divisor > 0, "divisor must be positive for ", name);
    TORCH_CHECK(value % divisor == 0, name, " must be divisible by tensor parallel size");
    return value / divisor;
}

}  // namespace sglang
