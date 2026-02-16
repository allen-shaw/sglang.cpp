cmake_minimum_required(VERSION 3.10)
project(sglang_cpp CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)

# Protobuf generation
set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/protos)
set(PROTO_HDR_DIR ${CMAKE_CURRENT_BINARY_DIR}/protos)
file(MAKE_DIRECTORY ${PROTO_HDR_DIR})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS 
    ${PROTO_SRC_DIR}/sglang.proto
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR} # For generated proto headers
    ${Protobuf_INCLUDE_DIRS}
)

add_executable(sglang_server
    src/main.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

target_link_libraries(sglang_server
    ${Protobuf_LIBRARIES}
    Threads::Threads
)
