cmake_minimum_required(VERSION 3.14)
project(sglang_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CUDA architecture to avoid auto-detection issues
set(TORCH_CUDA_ARCH_LIST "8.9")

# LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/torch;${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})

# Find dependencies
find_package(Threads REQUIRED)

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
)

add_library(sglang_core STATIC ${SGLANG_CORE_SRCS})

target_link_libraries(sglang_core PUBLIC
    ${TORCH_LIBRARIES}
    Threads::Threads
)

target_include_directories(sglang_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
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

# Legacy header compilation test
add_executable(test_core_headers tests/core/test_headers.cpp)
target_link_libraries(test_core_headers PRIVATE sglang_core)
