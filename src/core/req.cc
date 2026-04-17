#include "sglang/core/req.h"

namespace sglang {

void Req::initialize_runtime_state() {
  if (device_len_ < 0) {
    device_len_ = static_cast<int>(input_ids.size(0));
  }
  if (max_device_len_ < 0) {
    max_device_len_ = device_len_ + output_len;
  }
  validate_runtime_state();
}

void Req::validate_runtime_state() const {
  TORCH_CHECK(input_ids.dim() == 1, "Req::input_ids must be a 1D tensor");
  TORCH_CHECK(input_ids.device().is_cpu(), "Req::input_ids must stay on CPU");
  TORCH_CHECK(device_len_ >= 0, "Req::device_len must be initialized");
  TORCH_CHECK(max_device_len_ >= device_len_,
              "Req::max_device_len must be >= device_len");
  TORCH_CHECK(cached_len >= 0 && cached_len <= device_len_,
              "Req::cached_len must be within [0, device_len]");
  TORCH_CHECK(input_ids.size(0) <= device_len_,
              "Req::input_ids length cannot exceed device_len");
}

int Req::device_len() const {
  return device_len_ >= 0 ? device_len_ : static_cast<int>(input_ids.size(0));
}

int Req::max_device_len() const {
  return max_device_len_ >= 0 ? max_device_len_ : device_len() + output_len;
}

int Req::remain_len() const {
  return max_device_len() - device_len();
}

int Req::extend_len() const {
  return device_len() - cached_len;
}

bool Req::can_decode() const {
  return !is_chunked_prefill && remain_len() > 0;
}

void Req::complete_one() {
  initialize_runtime_state();
  cached_len = device_len_;
  if (device_len_ < max_device_len_) {
    ++device_len_;
  }
}

void Req::append_host(const torch::Tensor& next_token) {
  TORCH_CHECK(next_token.dim() == 1, "Req::append_host expects a 1D tensor");
  TORCH_CHECK(next_token.device().is_cpu(),
              "Req::append_host expects a CPU tensor");
  if (device_len_ < 0) {
    initialize_runtime_state();
  }
  TORCH_CHECK(input_ids.size(0) + next_token.size(0) <= device_len_,
              "Req::append_host cannot advance beyond device_len");
  input_ids = torch::cat({input_ids, next_token.to(input_ids.device())});
  TORCH_CHECK(input_ids.size(0) <= device_len_,
              "Req::append_host cannot advance beyond device_len");
}

std::string Req::toString() const {
  std::ostringstream oss;
  oss << "Req(table_idx=" << table_idx
      << ", cached_len=" << cached_len
      << ", device_len=" << device_len()
      << ", max_device_len=" << max_device_len()
      << ", is_chunked_prefill=" << (is_chunked_prefill ? "true" : "false")
      << ")";
  return oss.str();
}

} // namespace sglang
