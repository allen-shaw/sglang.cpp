cmake_minimum_required(VERSION 3.10)
project(sglang_cpp CXX)

set(CMAKE_CXX_STANDARD 23)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Find dependencies
# Find dependencies
set(Protobuf_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/protobuf/lib/cmake/protobuf")
find_package(Protobuf CONFIG REQUIRED)
set(Protobuf_PROTOC_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/protobuf/bin/protoc")
find_package(Threads REQUIRED)

# Protobuf generation
# Protobuf generation
set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_HDR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
# file(MAKE_DIRECTORY ${PROTO_HDR_DIR}) # Directory already exists in source

add_custom_command(
    OUTPUT ${PROTO_HDR_DIR}/sglang.pb.h ${PROTO_HDR_DIR}/sglang.pb.cc
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --cpp_out ${CMAKE_CURRENT_SOURCE_DIR} -I ${CMAKE_CURRENT_SOURCE_DIR} ${PROTO_SRC_DIR}/sglang.proto
    DEPENDS ${PROTO_SRC_DIR}/sglang.proto
)

set(PROTO_SRCS ${PROTO_HDR_DIR}/sglang.pb.cc)
set(PROTO_HDRS ${PROTO_HDR_DIR}/sglang.pb.h)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR} # For generated proto headers
    ${Protobuf_INCLUDE_DIRS}
)

add_executable(sglang_server
    src/main.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

target_link_libraries(sglang_server
    protobuf::libprotobuf
    ${TORCH_LIBRARIES}
    Threads::Threads
)

# LibTorch
# LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/torch;${CMAKE_PREFIX_PATH}")
# LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/torch;${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})

# Use Torch's protobuf if available, or ensure consistency
# For now, we rely on Torch's include dirs being first if they contain protobuf
# And we might need to link against Torch's protobuf explicitly if FindProtobuf picks system one
# But usually defining ABI=0 is the main fix if we link correctly.


# Test target for headers
add_executable(test_core_headers
    tests/core/test_headers.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

target_link_libraries(test_core_headers
    ${TORCH_LIBRARIES}
    protobuf::libprotobuf
    Threads::Threads
)

target_include_directories(test_core_headers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
