#include "gpt-service.hpp"

int main(void)
{

    std::string server_address("0.0.0.0:50051");
    GptService service;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    server->Wait();

    return 0;
}

