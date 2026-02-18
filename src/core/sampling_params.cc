#include "sglang/core/sampling_params.h"

namespace sglang {

bool SamplingParams::is_greedy() const {
  constexpr float kEpsilon = 1e-6F;
  if (temperature < kEpsilon) {
    return true;
  }
  if (top_k == 1) {
    return true;
  }
  return false;
}

} // namespace sglang
