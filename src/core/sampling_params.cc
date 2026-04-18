#include "sglang/core/sampling_params.h"

namespace sglang {

bool SamplingParams::is_greedy() const {
  constexpr float kEpsilon = 1e-6F;
  const bool greedy_temperature = temperature <= kEpsilon;
  const bool greedy_topk = top_k == 1;
  return (greedy_temperature || greedy_topk) && top_p >= 1.0F - kEpsilon;
}

} // namespace sglang
