
# NOTE: Latest versions of protobuf use `abseil-cpp` which requires >= c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(TARGET main_grpc)
set(grpc_cpp_plugin_location "/usr/bin/grpc_cpp_plugin")

set(PROTO_FILES gpt.proto)

add_executable(${TARGET} main.cpp gpt-service.cpp)

protobuf_generate(
    TARGET ${TARGET}
    LANGUAGE cpp
    PROTOS ${PROTO_FILES}
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}"
)

protobuf_generate(
    TARGET ${TARGET}
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}"
    PROTOS ${PROTO_FILES}
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}"
)

target_include_directories(${TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/")
target_link_libraries(${TARGET} PRIVATE common llama gRPC::grpc++ protobuf::libprotobuf ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
