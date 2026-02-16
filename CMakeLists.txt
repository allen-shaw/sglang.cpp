cmake_minimum_required(VERSION 3.10)
project(sglang_cpp CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Find dependencies
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)

# Protobuf generation
# Protobuf generation
set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_HDR_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto)
file(MAKE_DIRECTORY ${PROTO_HDR_DIR})

add_custom_command(
    OUTPUT ${PROTO_HDR_DIR}/sglang.pb.h ${PROTO_HDR_DIR}/sglang.pb.cc
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} -I ${CMAKE_CURRENT_SOURCE_DIR} ${PROTO_SRC_DIR}/sglang.proto
    DEPENDS ${PROTO_SRC_DIR}/sglang.proto
)

set(PROTO_SRCS ${PROTO_HDR_DIR}/sglang.pb.cc)
set(PROTO_HDRS ${PROTO_HDR_DIR}/sglang.pb.h)

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
