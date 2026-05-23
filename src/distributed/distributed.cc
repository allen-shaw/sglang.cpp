#include "sglang/distributed/distributed.h"

#include <ATen/cuda/CUDAContext.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

namespace sglang {
namespace {

std::mutex g_tp_mutex;
std::optional<TensorParallelInfo> g_test_tp_info;

struct ProcessGroupState {
    TensorParallelInfo info;
    std::string id_file;
    int device_index = -1;
    ncclComm_t comm = nullptr;
};

std::mutex g_nccl_mutex;
std::optional<ProcessGroupState> g_nccl_state;

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

std::string nccl_id_file_from_env() {
    const std::string master_addr = master_addr_from_env();
    const int master_port = master_port_from_env();
    std::ostringstream fallback;
    fallback << "/tmp/sglang_tp_nccl_" << master_addr << "_" << master_port << ".id";
    std::string path = fallback.str();
    for (char& ch : path) {
        if (ch == ':' || ch == '/') {
            ch = '_';
        }
    }
    return read_env_string("SGLANG_TP_NCCL_ID_FILE", path);
}

void check_nccl(ncclResult_t result, const char* operation) {
    TORCH_CHECK(result == ncclSuccess, operation, " failed: ", ncclGetErrorString(result));
}

void check_cuda(cudaError_t result, const char* operation) {
    TORCH_CHECK(result == cudaSuccess, operation, " failed: ", cudaGetErrorString(result));
}

void write_nccl_id_file(const std::string& path, const ncclUniqueId& id) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        TORCH_CHECK(out.good(), "Failed to open NCCL unique id file for write: ", tmp);
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        TORCH_CHECK(out.good(), "Failed to write NCCL unique id file: ", tmp);
    }
    TORCH_CHECK(std::rename(tmp.c_str(), path.c_str()) == 0,
                "Failed to publish NCCL unique id file: ", path);
}

ncclUniqueId read_nccl_id_file(const std::string& path, int timeout_sec) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (std::chrono::steady_clock::now() < deadline) {
        std::ifstream in(path, std::ios::binary);
        if (in.good()) {
            ncclUniqueId id;
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (in.gcount() == static_cast<std::streamsize>(sizeof(id))) {
                return id;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    TORCH_CHECK(false, "Timed out waiting for NCCL unique id file: ", path);
}

ncclDataType_t nccl_dtype_for(const torch::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case torch::kFloat32:
            return ncclFloat32;
        case torch::kFloat16:
            return ncclFloat16;
        case torch::kBFloat16:
            return ncclBfloat16;
        case torch::kFloat64:
            return ncclFloat64;
        case torch::kInt32:
            return ncclInt32;
        case torch::kInt64:
            return ncclInt64;
        default:
            TORCH_CHECK(false, "Unsupported tensor dtype for NCCL all-reduce: ",
                        tensor.scalar_type());
    }
}

ProcessGroupState create_nccl_state(const TensorParallelInfo& info,
                                    int device_index,
                                    const std::string& id_file) {
    const int timeout_sec = read_env_int("SGLANG_TP_TIMEOUT_SEC", 600);
    TORCH_CHECK(timeout_sec > 0, "SGLANG_TP_TIMEOUT_SEC must be positive");

    ncclUniqueId id;
    if (info.rank == 0) {
        check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
        write_nccl_id_file(id_file, id);
    } else {
        id = read_nccl_id_file(id_file, timeout_sec);
    }

    check_cuda(cudaSetDevice(device_index), "cudaSetDevice");
    ncclComm_t comm = nullptr;
    check_nccl(ncclCommInitRank(&comm, info.size, id, info.rank), "ncclCommInitRank");
    return ProcessGroupState{info, id_file, device_index, comm};
}

ncclComm_t get_or_create_nccl_comm(const TensorParallelInfo& info, int device_index) {
    std::lock_guard<std::mutex> lock(g_nccl_mutex);
    const std::string id_file = nccl_id_file_from_env();
    if (!g_nccl_state.has_value() ||
        g_nccl_state->info.rank != info.rank ||
        g_nccl_state->info.size != info.size ||
        g_nccl_state->device_index != device_index ||
        g_nccl_state->id_file != id_file) {
        g_nccl_state = create_nccl_state(info, device_index, id_file);
    }
    return g_nccl_state->comm;
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
    const int device_index = out.device().index();
    ncclComm_t comm = get_or_create_nccl_comm(info, device_index);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    check_nccl(ncclAllReduce(out.data_ptr(),
                             out.data_ptr(),
                             static_cast<size_t>(out.numel()),
                             nccl_dtype_for(out),
                             ncclSum,
                             comm,
                             stream),
               "ncclAllReduce");
    return out;
}

int divide_even(int value, int divisor, const char* name) {
    TORCH_CHECK(divisor > 0, "divisor must be positive for ", name);
    TORCH_CHECK(value % divisor == 0, name, " must be divisible by tensor parallel size");
    return value / divisor;
}

}  // namespace sglang
