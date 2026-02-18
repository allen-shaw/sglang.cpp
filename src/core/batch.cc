#include "sglang/core/batch.h"

namespace sglang {

bool Batch::is_prefill() const {
  return phase == BatchPhase::Prefill;
}

bool Batch::is_decode() const {
  return phase == BatchPhase::Decode;
}

int Batch::size() const {
  return reqs.size();
}

int Batch::padded_size() const {
  return padded_reqs.size();
}

} // namespace sglang
