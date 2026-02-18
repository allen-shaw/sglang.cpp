cmake_minimum_required(VERSION 3.10)
project(sglang_cpp CXX)

set(CMAKE_CXX_STANDARD 23)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/torch;${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})

# Find dependencies
find_package(Threads REQUIRED)

# Core sources
set(SGLANG_CORE_SRCS
    src/core/req.cc
    src/core/batch.cc
    src/core/context.cc
    src/core/sampling_params.cc
    src/utils/tensor_utils.cc
)

add_executable(sglang_server
    src/main.cpp
    ${SGLANG_CORE_SRCS}
)

target_link_libraries(sglang_server
    ${TORCH_LIBRARIES}
    Threads::Threads
)

target_include_directories(sglang_server PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Use Torch's protobuf if available, or ensure consistency
# For now, we rely on Torch's include dirs being first if they contain protobuf
# And we might need to link against Torch's protobuf explicitly if FindProtobuf picks system one
# But usually defining ABI=0 is the main fix if we link correctly.


# Test target for headers
add_executable(test_core_headers
    tests/core/test_headers.cpp
)

target_link_libraries(test_core_headers
    ${TORCH_LIBRARIES}
    Threads::Threads
)

target_include_directories(test_core_headers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
