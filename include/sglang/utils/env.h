#pragma once

#include <cstdlib>
#include <optional>
#include <string>

namespace sglang {

/// Environment configuration singleton, matching Python EnvClassSingleton.
/// Reads environment variables with SGLANG_ prefix.
class EnvConfig {
 public:
  static EnvConfig& instance() {
    static EnvConfig inst;
    return inst;
  }

  // Shell defaults
  int shell_max_tokens = 2048;
  int shell_top_k = -1;
  float shell_top_p = 1.0f;
  float shell_temperature = 0.6f;

  // Backend runtime
  std::optional<bool> flashinfer_use_tensor_cores = std::nullopt;
  bool disable_overlap_scheduling = false;
  int64_t pynccl_max_buffer_size = 1024LL * 1024 * 1024;  // 1GB

 private:
  EnvConfig() {
    read_int("SGLANG_SHELL_MAX_TOKENS", shell_max_tokens);
    read_int("SGLANG_SHELL_TOP_K", shell_top_k);
    read_float("SGLANG_SHELL_TOP_P", shell_top_p);
    read_float("SGLANG_SHELL_TEMPERATURE", shell_temperature);
    read_bool("SGLANG_DISABLE_OVERLAP_SCHEDULING", disable_overlap_scheduling);

    if (const char* v = std::getenv("SGLANG_FLASHINFER_USE_TENSOR_CORES")) {
      flashinfer_use_tensor_cores = to_bool(v);
    }
  }

  static bool to_bool(const char* s) {
    std::string v(s);
    return v == "1" || v == "true" || v == "True" || v == "yes";
  }

  static void read_int(const char* name, int& target) {
    if (const char* v = std::getenv(name)) {
      try { target = std::stoi(v); } catch (...) {}
    }
  }

  static void read_float(const char* name, float& target) {
    if (const char* v = std::getenv(name)) {
      try { target = std::stof(v); } catch (...) {}
    }
  }

  static void read_bool(const char* name, bool& target) {
    if (const char* v = std::getenv(name)) {
      target = to_bool(v);
    }
  }
};

}  // namespace sglang
