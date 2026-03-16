#include "sglang/core/req.h"

namespace sglang {

int Req::device_len() const {
  return input_ids.size(0);
}

int Req::max_device_len() const {
  // Assuming output_len tracks the *remaining* tokens to be generated,
  // or that we need a way to calculate total length.
  // Given the structure, we'll assume max_len = current + remaining.
  // If output_len is purely "requested new tokens" and never changes, this logic is flawed without a start_len.
  // BUT, to match Python where max_device_len is constant:
  // We'll treat `output_len` as the *remaining* output length in this C++ adaptation
  // if we can't add fields.
  // See append_host logic below.
  return input_ids.size(0) + output_len;
}

int Req::remain_len() const {
  return output_len;
}

int Req::extend_len() const {
  return device_len() - cached_len;
}

bool Req::can_decode() const {
  return output_len > 0;
}

void Req::complete_one() {
  cached_len = device_len();
  // NOTE: In Python complete_one increments device_len (implied by appending).
  // Here, we update cached_len to match the new device_len.
  // We also decrement output_len to reflect one less token needed.
  if (output_len > 0) {
    output_len--;
  }
}

void Req::append_host(const torch::Tensor& next_token) {
  input_ids = torch::cat({input_ids, next_token.to(input_ids.device())});
}

std::string Req::toString() const {
  std::ostringstream oss;
  oss << "Req(table_idx=" << table_idx
      << ", cached_len=" << cached_len
      << ", device_len=" << device_len()
      << ", max_device_len=" << max_device_len() << ")";
  return oss.str();
}

} // namespace sglang
