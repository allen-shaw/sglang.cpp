#pragma once

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>

namespace sglang {

enum class LogLevel : int {
  kDebug = 0,
  kInfo = 1,
  kWarning = 2,
  kError = 3,
  kCritical = 4,
};

namespace detail {

inline LogLevel get_log_level() {
  static LogLevel level = [] {
    const char* env = std::getenv("SGLANG_LOG_LEVEL");
    if (!env) return LogLevel::kInfo;
    std::string s(env);
    if (s == "DEBUG") return LogLevel::kDebug;
    if (s == "WARNING") return LogLevel::kWarning;
    if (s == "ERROR") return LogLevel::kError;
    if (s == "CRITICAL") return LogLevel::kCritical;
    return LogLevel::kInfo;
  }();
  return level;
}

inline const char* level_string(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:    return "DEBUG   ";
    case LogLevel::kInfo:     return "INFO    ";
    case LogLevel::kWarning:  return "WARNING ";
    case LogLevel::kError:    return "ERROR   ";
    case LogLevel::kCritical: return "CRITICAL";
  }
  return "UNKNOWN ";
}

inline const char* level_color(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:    return "\033[36m";  // Cyan
    case LogLevel::kInfo:     return "\033[32m";  // Green
    case LogLevel::kWarning:  return "\033[33m";  // Yellow
    case LogLevel::kError:    return "\033[31m";  // Red
    case LogLevel::kCritical: return "\033[35m";  // Magenta
  }
  return "";
}

inline std::string timestamp() {
  std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d|%H:%M:%S", std::localtime(&t));
  return buf;
}

inline std::mutex& log_mutex() {
  static std::mutex mtx;
  return mtx;
}

inline void log_message(LogLevel level, std::string_view file, int line,
                        std::string_view msg) {
  if (level < get_log_level()) return;
  
  const char* reset = "\033[0m";
  const char* bold = "\033[1m";
  
  std::lock_guard lock(log_mutex());
  std::cerr << bold << "[" << timestamp() << "] " << reset
            << level_color(level) << level_string(level) << reset
            << " " << msg << "\n";
}

}  // namespace detail
}  // namespace sglang

#define SGLANG_LOG(level, msg)                                          \
  do {                                                                  \
    ::sglang::detail::log_message(level, __FILE__, __LINE__, msg);      \
  } while (0)

#define SGLANG_LOG_DEBUG(msg)   SGLANG_LOG(::sglang::LogLevel::kDebug, msg)
#define SGLANG_LOG_INFO(msg)    SGLANG_LOG(::sglang::LogLevel::kInfo, msg)
#define SGLANG_LOG_WARN(msg)    SGLANG_LOG(::sglang::LogLevel::kWarning, msg)
#define SGLANG_LOG_ERROR(msg)   SGLANG_LOG(::sglang::LogLevel::kError, msg)
#define SGLANG_LOG_CRITICAL(msg) SGLANG_LOG(::sglang::LogLevel::kCritical, msg)
