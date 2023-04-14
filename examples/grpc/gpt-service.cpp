#include <sstream>
#include "gpt-service.hpp"


grpc::Status process_embd(std::vector<llama_token>& embd, int& n_past, const std::vector<llama_token>& last_n_tokens, int length_of_ctx, llama_context* ctx, const gpt_params& params) {
    if (embd.size() > 0) {
        // infinite text generation via context swapping
        // if we run out of context:
        // - take the n_keep first tokens from the original prompt (via n_past)
        // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
        if (n_past + (int) embd.size() > length_of_ctx) {
            const int n_left = n_past - params.n_keep;

            n_past = params.n_keep;

            // insert n_left/2 tokens at the start of embd from last_n_tokens
            embd.insert(embd.begin(), last_n_tokens.begin() + length_of_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());
        }

        if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads))
        {
            std::ostringstream oss;
            oss << __func__ << ": failed to eval\n";
            return grpc::Status(grpc::StatusCode::INTERNAL, oss.str());
        }
    }

    return grpc::Status::OK;
}

void process_input(std::vector<llama_token>& embd, int& n_remain, int& n_consumed, bool& input_noecho, std::vector<llama_token>& last_n_tokens, int length_of_ctx, llama_context* ctx, const gpt_params& params, std::vector<llama_token>& input_embedings, const std::vector<llama_token>& newline_token, bool is_interacting) {
    if ((int) input_embedings.size() <= n_consumed && !is_interacting) {
        // out of user input, sample next token
        const int32_t top_k          = params.top_k;
        const float   top_p          = params.top_p;
        const float   temp           = params.temp;
        const float   repeat_penalty = params.repeat_penalty;

        llama_token id = 0;

        {
            auto logits = llama_get_logits(ctx);

            if (params.ignore_eos) {
                logits[llama_token_eos()] = 0;
            }

            id = llama_sample_top_p_top_k(ctx,
                    last_n_tokens.data() + length_of_ctx - params.repeat_last_n,
                    params.repeat_last_n, top_k, top_p, temp, repeat_penalty);

            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(id);
        }

        // replace end of text token with newline token when in interactive mode
        if (id == llama_token_eos() && params.interactive && !params.instruct) {
            id = newline_token.front();
            if (params.antiprompt.size() != 0) {
                // tokenize and inject first reverse prompt
                const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false);
                input_embedings.insert(input_embedings.end(), first_antiprompt.begin(), first_antiprompt.end());
            }
        }

        // add it to the context
        embd.push_back(id);

        // echo this to console
        input_noecho = false;

        // decrement remaining sampling budget
        --n_remain;
    } else {
        // some user input remains from prompt or interaction, forward it to processing
        while ((int) input_embedings.size() > n_consumed) {
            embd.push_back(input_embedings[n_consumed]);
            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(input_embedings[n_consumed]);
            ++n_consumed;
            if ((int) embd.size() >= params.n_batch) {
                break;
            }
        }
    }
}

grpc::Status GptService::RunGpt(grpc::ServerContext* context, const gpt::GptParams* request, gpt::SetupDone* response)
{
    this->params.n_threads = request->n_threads();
    this->params.n_predict = request->n_predict();
    this->params.repeat_last_n = request->repeat_last_n();
    this->params.n_batch = request->n_batch();
    this->params.n_keep = request->n_keep();

    // sampling parameters
    this->params.top_k = request->top_k();
    this->params.top_p = request->top_p();
    this->params.temp = request->temp();
    this->params.repeat_penalty = request->repeat_penalty();

    this->params.model = request->model();
    this->params.prompt = request->prompt();
    this->params.input_prefix = request->input_prefix();

    std::copy(request->antiprompt().begin(), request->antiprompt().end(), std::back_inserter(this->params.antiprompt));

    this->params.random_prompt = request->random_prompt();
    this->params.use_color = request->use_color();
    this->params.interactive = request->interactive();
    this->params.embedding = request->embedding();
    this->params.interactive_start = request->interactive_start();

    this->params.instruct = request->instruct();
    this->params.ignore_eos = request->ignore_eos();
    this->params.perplexity = request->perplexity();
    // this->params.mem_test = request->mem_test();
    this->params.verbose_prompt = request->verbose_prompt();


    // load the model
    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = request->n_ctx();
        lparams.n_parts    = request->n_parts();
        lparams.seed       = request->seed();
        lparams.f16_kv     = request->memory_f16();
        lparams.use_mmap   = request->use_mmap();
        lparams.use_mlock  = request->use_mlock();

        this->ctx = llama_init_from_file(request->model().c_str(), lparams);

        if (this->ctx == NULL) {
            std::ostringstream oss;
            oss << __func__ << ": error: failed to load model '" << params.model << "'\n";
            return grpc::Status(grpc::StatusCode::INTERNAL, oss.str());
        }
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    this->params.prompt.insert(0, 1, ' ');


    // tokenize the prompt
    this->embd_inp = ::llama_tokenize(this->ctx, this->params.prompt, true);

    this->n_ctx = llama_n_ctx(this->ctx);

    if ((int) this->embd_inp.size() > this->n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) this->embd_inp.size(), this->n_ctx - 4);
        return grpc::Status(grpc::StatusCode::INTERNAL, "prompt is too long");
    }

    // // number of tokens to keep when resetting context
    if (this->params.n_keep < 0 || this->params.n_keep > (int)this->embd_inp.size() || this->params.instruct) {
        this->params.n_keep = (int)this->embd_inp.size();
    }

    // prefix & suffix for instruct mode
    this->inp_pfx = ::llama_tokenize(this->ctx, "\n\n### Instruction:\n\n", true);
    this->inp_sfx = ::llama_tokenize(this->ctx, "\n\n### Response:\n\n", false);

    // in instruct mode, we inject a prefix and a suffix to each input by the user
    if (this->params.instruct) {
        this->params.interactive_start = true;
        this->params.antiprompt.push_back("### Instruction:\n\n");
    }

    // enable interactive mode if reverse prompt or interactive start is specified
    if (this->params.antiprompt.size() != 0 || this->params.interactive_start) {
        this->params.interactive = true;
    }

    // determine newline token
    this->llama_token_newline = ::llama_tokenize(this->ctx, "\n", false);

    if (this->params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, this->params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, this->embd_inp.size());
        for (int i = 0; i < (int) this->embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n", this->embd_inp[i], llama_token_to_str(this->ctx, this->embd_inp[i]));
        }
        if (this->params.n_keep > 0) {
        fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < this->params.n_keep; i++) {
                fprintf(stderr, "%s", llama_token_to_str(this->ctx, this->embd_inp[i]));
            }
            fprintf(stderr, "'\n");
        }
        fprintf(stderr, "\n");
    }

    if (this->params.interactive)
    {
        fprintf(stderr, "%s: interactive mode on.\n", __func__);

        if (this->params.antiprompt.size())
        {
            for (auto antiprompt : this->params.antiprompt)
            {
                fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str());
            }
        }

        if (!this->params.input_prefix.empty())
        {
            fprintf(stderr, "Input prefix: '%s'\n", this->params.input_prefix.c_str());
        }
    }

    fprintf(stderr, "sampling: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n",
        this->params.temp, this->params.top_k, this->params.top_p, this->params.repeat_last_n, this->params.repeat_penalty);
    fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", this->n_ctx, this->params.n_batch, this->params.n_predict, this->params.n_keep);
    fprintf(stderr, "\n\n");

    std::string msg("done");

    response->set_instance_id(msg);
    return grpc::Status::OK;
}

grpc::Status GptService::AskGpt(grpc::ServerContext* context, const gpt::GptQuery* request, grpc::ServerWriter<gpt::GptAnswer>* writer)
{
    std::ostringstream oss;
    oss << " " << request->prompt();

    // tokenize the prompt
    std::cout << "P >>> " << request->prompt() << std::endl;
    auto input_embedings = ::llama_tokenize(this->ctx, oss.str(), true);

    // Get length of context
    auto length_of_ctx = llama_n_ctx(this->ctx);

    // Get length of prompt
    auto length_of_prompt =  input_embedings.size();

    if (length_of_prompt > length_of_ctx - 4)
    {
        // 4 - it's some magic number idk.
        fprintf(stderr, "%s: error: prompt is too long (%lu tokens, max %d)\n", __func__, length_of_prompt, length_of_ctx - 4);
        return grpc::Status(grpc::StatusCode::INTERNAL, "prompt is too long");
    }

    // enable interactive mode if reverse prompt or interactive start is specified
    if (this->params.antiprompt.size() != 0 || this->params.interactive_start) {
        this->params.interactive = true;
    }

    // determine newline token
    auto newline_token = ::llama_tokenize(this->ctx, "\n", false);
    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(length_of_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    if (this->params.interactive) {
        fprintf(stderr, "== Running in interactive mode. ==\n");
        this->is_interacting = this->params.interactive_start;
    }

    std::vector<llama_token> embd;

    bool is_antiprompt = false;
    bool input_noecho  = false;

    int n_past     = 0;
    int n_remain   = this->params.n_predict;
    int n_consumed = 0;


    //========================
    // predict

    while (n_remain != 0 || params.interactive) {
        auto status = process_embd(embd, n_past, last_n_tokens, length_of_ctx, this->ctx, this->params);
        if (!status.ok()) {
            return status;
        }

        n_past += embd.size();
        embd.clear();

        process_input(embd, n_remain, n_consumed, input_noecho, last_n_tokens, length_of_ctx, this->ctx, this->params, input_embedings, this->llama_token_newline, this->is_interacting);

        if (!input_noecho) {
            gpt::GptAnswer answer;
            for (auto id : embd) {
                auto token_as_str = llama_token_to_str(this->ctx, id);
                printf("%s", token_as_str);
                answer.set_message(token_as_str);
                writer->Write(answer);
            }
            fflush(stdout);
        }

        // grpc::Status status2 = this->handle_interactive_mode(writer, is_antiprompt, input_embedings, n_consumed, n_remain, input_noecho, n_past, last_n_tokens);
        if (this->params.interactive && static_cast<int>(input_embedings.size()) <= n_consumed) {

            // check for reverse prompt
            if (this->params.antiprompt.size()) {
                std::string last_output;
                for (auto id : last_n_tokens) {
                    last_output += llama_token_to_str(this->ctx, id);
                }

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                for (std::string antiprompt : this->params.antiprompt) {
                    if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                        this->is_interacting = true;
                        is_antiprompt = true;
                        break;
                    }
                }
            }

            if (n_past > 0 && this->is_interacting) {
                std::string buffer;
                if (!this->params.input_prefix.empty()) {
                    gpt::GptAnswer answer;

                    buffer += this->params.input_prefix;
                    printf("%s", buffer.c_str());

                    answer.set_message("[EOI]");
                    writer->Write(answer);

                    return grpc::Status::OK;
                }

                if (buffer.length() > 1) {
                    auto line_inp = ::llama_tokenize(this->ctx, buffer, false);
                    input_embedings.insert(input_embedings.end(), line_inp.begin(), line_inp.end());

                    n_remain -= line_inp.size();
                }

                input_noecho = true;
            }

            if (n_past > 0) {
                this->is_interacting = false;
            }
        }

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos()) {
            gpt::GptAnswer answer;
            if (this->params.instruct) {
                answer.set_message("[instruct][end of text]");
                writer->Write(answer);
                this->is_interacting = true;
            } else {
                answer.set_message("[end of text]");
                writer->Write(answer);
                fprintf(stderr, " [end of text]\n");
                break;
            }
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (this->params.interactive && n_remain <= 0 && this->params.n_predict != -1) {
            n_remain = this->params.n_predict;
            this->is_interacting = true;
        }
    }

    //========================
    return grpc::Status::OK;
}
