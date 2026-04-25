cmake_minimum_required(VERSION 3.14)
project(sglang_cpp CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CUDA architecture to avoid auto-detection issues
set(TORCH_CUDA_ARCH_LIST "8.9")

# LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/torch;/home/allen/miniconda3/lib/python3.12/site-packages/torch/share/cmake;${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find dependencies
find_package(Threads REQUIRED)
include(FetchContent)

# ============================================================
# Cinatra (header-only HTTP server)
# ============================================================
FetchContent_Declare(
    cinatra
    GIT_REPOSITORY https://github.com/qicosmos/cinatra.git
    GIT_TAG        0.9.4
)
FetchContent_GetProperties(cinatra)
if(NOT cinatra_POPULATED)
  FetchContent_Populate(cinatra)
endif()

# ============================================================
# FlashInfer (fetched via FetchContent)
# ============================================================
FetchContent_Declare(
    flashinfer
    GIT_REPOSITORY https://github.com/flashinfer-ai/flashinfer.git
    GIT_TAG        v0.2.0.post1
)
FetchContent_GetProperties(flashinfer)
if(NOT flashinfer_POPULATED)
  FetchContent_Populate(flashinfer)
endif()
# ============================================================
# tokenizers-cpp (fetched via FetchContent)
# ============================================================
FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        main
)
FetchContent_GetProperties(tokenizers_cpp)
if(NOT tokenizers_cpp_POPULATED)
  FetchContent_Populate(tokenizers_cpp)
  add_subdirectory(${tokenizers_cpp_SOURCE_DIR} ${tokenizers_cpp_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
# ============================================================
# nlohmann/json (header-only, for config parsing)
# ============================================================
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# ============================================================
# Google Benchmark
# ============================================================
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.4
)
FetchContent_MakeAvailable(google_benchmark)

# ============================================================
# Core static library (shared by server and tests)
# ============================================================
set(SGLANG_CORE_SRCS
    src/core/req.cc
    src/core/batch.cc
    src/core/context.cc
    src/core/sampling_params.cc
    src/utils/tensor_utils.cc
    src/message/message.cc
    src/kernels/radix.cc
    src/kernels/token_pool.cu
    src/kvcache/radix_tree.cc
    src/kvcache/cache_manager.cc
    src/kvcache/mha_kvcache.cu
    src/layers/activation.cu
    src/layers/normalization.cu
    src/layers/rotary_embedding.cu
    src/layers/linear.cc
    src/layers/embedding.cc
    src/layers/attention_layer.cc
    src/attention/flashinfer_backend.cu
    src/tokenizer/hf_tokenizer.cc
    src/models/utils.cc
    src/models/llama.cc
    src/models/qwen.cc
    src/models/qwen3.cc
    src/models/mistral.cc
    src/models/qwen3_moe.cc
    src/models/config.cc
    src/models/weight_loader.cc
    src/moe/moe.cc
    src/engine/config.cc
    src/engine/sampling.cu
    src/engine/engine.cc
    src/scheduler/prefill.cc
    src/scheduler/decode.cc
    src/scheduler/scheduler.cc
    src/llm/llm.cc
    src/server/args.cc
    src/server/frontend.cc
    src/server/http_server.cc
)

add_library(sglang_core STATIC ${SGLANG_CORE_SRCS})

target_compile_definitions(sglang_core PUBLIC
    CINATRA_LOG_DEBUG=cinatra::NULL_LOGGER
    CINATRA_LOG_TRACE=cinatra::NULL_LOGGER
)

target_link_libraries(sglang_core PUBLIC
    ${TORCH_LIBRARIES}
    tokenizers_cpp
    nlohmann_json::nlohmann_json
    Threads::Threads
)

target_include_directories(sglang_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${flashinfer_SOURCE_DIR}/include
    ${tokenizers_cpp_SOURCE_DIR}/include
    ${cinatra_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/safetensors_cpp
)

# ============================================================
# Main server executable
# ============================================================
add_executable(sglang_server src/main.cpp)

target_link_libraries(sglang_server PRIVATE sglang_core)

# ============================================================
# GTest (fetched via FetchContent)
# ============================================================
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ============================================================
# Test targets
# ============================================================

# Core tests
add_executable(test_req tests/core/test_req.cpp)
target_link_libraries(test_req PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestReq COMMAND test_req)

add_executable(test_batch tests/core/test_batch.cpp)
target_link_libraries(test_batch PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestBatch COMMAND test_batch)

add_executable(test_context tests/core/test_context.cpp)
target_link_libraries(test_context PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestContext COMMAND test_context)

add_executable(test_sampling_params tests/core/test_sampling_params.cpp)
target_link_libraries(test_sampling_params PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestSamplingParams COMMAND test_sampling_params)

# Utils tests
add_executable(test_tensor_utils tests/utils/test_tensor_utils.cpp)
target_link_libraries(test_tensor_utils PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestTensorUtils COMMAND test_tensor_utils)

# Message tests
add_executable(test_message tests/message/test_message.cpp)
target_link_libraries(test_message PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestMessage COMMAND test_message)

# KV Cache tests
add_executable(test_radix_tree tests/kvcache/test_radix_tree.cpp)
target_link_libraries(test_radix_tree PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestRadixTree COMMAND test_radix_tree)

add_executable(test_cache_manager tests/kvcache/test_cache_manager.cpp)
target_link_libraries(test_cache_manager PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestCacheManager COMMAND test_cache_manager)

# Kernel tests
add_executable(test_radix_kernel tests/kernels/test_radix.cpp)
target_link_libraries(test_radix_kernel PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestRadixKernel COMMAND test_radix_kernel)

# Legacy header compilation test
add_executable(test_core_headers tests/core/test_headers.cpp)
target_link_libraries(test_core_headers PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestCoreHeaders COMMAND test_core_headers)

# Phase 3.2 Tokenizer
add_executable(test_tokenizer tests/tokenizer/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestTokenizer COMMAND test_tokenizer)

# Phase 3
add_executable(test_activation tests/layers/test_activation.cpp)
target_link_libraries(test_activation PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestActivation COMMAND test_activation)

add_executable(test_normalization tests/layers/test_normalization.cpp)
target_link_libraries(test_normalization PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestNormalization COMMAND test_normalization)

add_executable(test_rotary_embedding tests/layers/test_rotary_embedding.cpp)
target_link_libraries(test_rotary_embedding PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestRotaryEmbedding COMMAND test_rotary_embedding)

add_executable(test_linear tests/layers/test_linear.cpp)
target_link_libraries(test_linear PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestLinear COMMAND test_linear)

add_executable(test_embedding tests/layers/test_embedding.cpp)
target_link_libraries(test_embedding PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestEmbedding COMMAND test_embedding)

add_executable(test_attention_layer tests/layers/test_attention_layer.cpp)
target_link_libraries(test_attention_layer PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestAttentionLayer COMMAND test_attention_layer)

add_executable(test_llama tests/models/test_llama.cpp)
target_link_libraries(test_llama PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestLlama COMMAND test_llama)

add_executable(test_mistral tests/models/test_mistral.cpp)
target_link_libraries(test_mistral PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestMistral COMMAND test_mistral)

add_executable(test_qwen tests/models/test_qwen.cpp)
target_link_libraries(test_qwen PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestQwen COMMAND test_qwen)

add_executable(test_qwen3_moe tests/models/test_qwen3_moe.cpp)
target_link_libraries(test_qwen3_moe PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestQwen3MoE COMMAND test_qwen3_moe)

add_executable(test_moe tests/moe/test_moe.cpp)
target_link_libraries(test_moe PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestMoE COMMAND test_moe)

add_executable(test_qwen3_integration tests/models/test_qwen3_integration.cpp)
target_link_libraries(test_qwen3_integration PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestQwen3Integration COMMAND test_qwen3_integration)

add_executable(test_sampler tests/engine/test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestSampler COMMAND test_sampler)

add_executable(test_engine_e2e tests/engine/test_engine_e2e.cpp)
target_link_libraries(test_engine_e2e PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestEngineE2E COMMAND test_engine_e2e)

add_executable(test_scheduler_managers tests/scheduler/test_managers.cpp)
target_link_libraries(test_scheduler_managers PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestSchedulerManagers COMMAND test_scheduler_managers)

add_executable(test_scheduler_e2e tests/scheduler/test_scheduler_e2e.cpp)
target_link_libraries(test_scheduler_e2e PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestSchedulerE2E COMMAND test_scheduler_e2e)

add_executable(test_llm tests/llm/test_llm.cpp)
target_link_libraries(test_llm PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestLLM COMMAND test_llm)

add_executable(test_server_args tests/server/test_args.cpp)
target_link_libraries(test_server_args PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestServerArgs COMMAND test_server_args)

add_executable(test_http_e2e tests/server/test_http_e2e.cpp)
target_link_libraries(test_http_e2e PRIVATE sglang_core GTest::gtest_main)
add_test(NAME TestHttpE2E COMMAND test_http_e2e)

# ============================================================
# Benchmark targets
# ============================================================
add_executable(bench_offline benchmarks/offline/bench_offline.cpp)
target_link_libraries(bench_offline PRIVATE sglang_core benchmark::benchmark)
