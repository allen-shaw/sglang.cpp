#include <iostream>
#include <string>
#include <vector>

#include "sglang/llm/llm.h"
#include "sglang/server/args.h"
#include "sglang/server/http_server.h"

int main(int argc, char** argv) {
  try {
    auto args = sglang::ServerArgsParser::parse(argc, argv);
    if (args.shell_mode) {
      sglang::LLM llm(args);
      sglang::SamplingParams sampling_params;
      sampling_params.temperature = 0.0F;
      sampling_params.top_p = 1.0F;
      sampling_params.top_k = -1;
      sampling_params.max_new_tokens = 128;

      std::string prompt;
      while (true) {
        std::cout << "> " << std::flush;
        if (!std::getline(std::cin, prompt)) {
          break;
        }
        if (prompt == "/exit") {
          break;
        }
        if (prompt.empty()) {
          continue;
        }
        auto result = llm.generate(prompt, sampling_params);
        std::cout << result.text << std::endl;
      }
      llm.shutdown();
      return 0;
    }

    sglang::ApiServer server(args);
    server.run();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "sglang_server failed: " << e.what() << std::endl;
    return 1;
  }
}
