#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>
#include <memory>
#include "gpt.pb.h"
#include "gpt.grpc.pb.h"
#include "llama.h"
#include "common.h"


class GptService final : public gpt::GptService::Service
{
  private:
    gpt_params params;

    int n_ctx;
    bool is_interacting = false;
    llama_context *ctx;
    std::vector<int> embd_inp;
    std::vector<int> llama_token_newline;
    std::vector<int> inp_pfx;
    std::vector<int> inp_sfx;

  public:
    grpc::Status RunGpt(grpc::ServerContext* context, const gpt::GptParams* request, gpt::SetupDone* response) override;
    grpc::Status AskGpt(grpc::ServerContext* context, const gpt::GptQuery* request, grpc::ServerWriter<gpt::GptAnswer>* writer) override;
    grpc::Status handle_interactive_mode(grpc::ServerWriter<gpt::GptAnswer>* writer, bool& is_antiprompt, std::vector<llama_token>& input_embedings, int& n_consumed, int& n_remain, bool& input_noecho, int n_past, const std::vector<llama_token>& last_n_tokens);
    ~GptService() = default;

};


