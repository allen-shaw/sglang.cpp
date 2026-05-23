#pragma once

#include <torch/torch.h>

namespace sglang {

struct TensorParallelInfo {
    int rank = 0;
    int size = 1;

    bool enabled() const {
        return size > 1;
    }
};

TensorParallelInfo get_tp_info();

int tp_rank();
int tp_size();

void set_tp_info_for_test(int rank, int size);
void reset_tp_info_for_test();

torch::Tensor tensor_model_parallel_all_reduce(const torch::Tensor& x);

int divide_even(int value, int divisor, const char* name);

}  // namespace sglang
