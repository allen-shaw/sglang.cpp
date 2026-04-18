#pragma once

#include <cassert>
#include <cstdint>
#include <string>

namespace sglang {

/// Divides a by b, asserting exact divisibility.
/// If allow_replicate=true, allows b > a when b % a == 0, returning 1.
inline int div_even(int a, int b, bool allow_replicate = false) {
  if (allow_replicate && b > a) {
    assert(b % a == 0 && "b must be divisible by a for KV head replication");
    return 1;
  }
  assert(a % b == 0 && "a must be divisible by b");
  return a / b;
}

/// Divides a by b, rounding up.
inline int div_ceil(int a, int b) {
  return (a + b - 1) / b;
}

/// Aligns a to the next multiple of b.
inline int align_ceil(int a, int b) {
  return div_ceil(a, b) * b;
}

/// Aligns a to the previous multiple of b.
inline int align_down(int a, int b) {
  return (a / b) * b;
}

/// Aligns a to the previous multiple of b (int64_t overload).
inline int64_t align_down(int64_t a, int64_t b) {
  return (a / b) * b;
}

inline int64_t align_down(int64_t a, int b) {
  return (a / static_cast<int64_t>(b)) * static_cast<int64_t>(b);
}

}  // namespace sglang
