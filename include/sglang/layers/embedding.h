#pragma once

#include <torch/torch.h>

namespace sglang {

class VocabParallelEmbedding {
 public:
    VocabParallelEmbedding(int num_embeddings, int embedding_dim);
    virtual ~VocabParallelEmbedding() = default;

    virtual torch::Tensor forward(const torch::Tensor& x);

    torch::Tensor weight;

 protected:
    int num_embeddings_;
    int num_embeddings_tp_;
    int tp_size_;
    int vocab_start_idx_;
    int vocab_end_idx_;
};

class ParallelLMHead : public VocabParallelEmbedding {
 public:
    ParallelLMHead(int num_embeddings, int embedding_dim, bool bias = false, VocabParallelEmbedding* tied_embedding = nullptr);

    torch::Tensor forward(const torch::Tensor& x) override;

    torch::Tensor bias;
    VocabParallelEmbedding* tied_embedding_;
};

}  // namespace sglang
