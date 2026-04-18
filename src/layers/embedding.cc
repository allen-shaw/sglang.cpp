#include "sglang/layers/embedding.h"

namespace sglang {

VocabParallelEmbedding::VocabParallelEmbedding(int num_embeddings, int embedding_dim)
    : num_embeddings_(num_embeddings) {
    tp_size_ = 1;
    num_embeddings_tp_ = (num_embeddings + tp_size_ - 1) / tp_size_;
    vocab_start_idx_ = num_embeddings_tp_ * 0;
    vocab_end_idx_ = std::min(vocab_start_idx_ + num_embeddings_tp_, num_embeddings);

    weight = register_parameter(
        "weight",
        torch::empty({num_embeddings_tp_, embedding_dim}, torch::device(torch::kCUDA).dtype(torch::kFloat16))
    );
}

torch::Tensor VocabParallelEmbedding::forward(const torch::Tensor& x) {
    return torch::nn::functional::embedding(x, weight);
}

ParallelLMHead::ParallelLMHead(int num_embeddings, int embedding_dim, bool has_bias, VocabParallelEmbedding* tied_embedding)
    : VocabParallelEmbedding(num_embeddings, embedding_dim), tied_embedding_(tied_embedding) {
    if (has_bias) {
        bias = register_parameter(
            "bias",
            torch::empty({num_embeddings_tp_}, torch::device(torch::kCUDA).dtype(torch::kFloat16))
        );
    }
}

torch::Tensor ParallelLMHead::forward(const torch::Tensor& x) {
    auto active_weight = tied_embedding_ ? tied_embedding_->weight : this->weight;
    return torch::nn::functional::linear(x, active_weight, this->bias.defined() ? this->bias : torch::Tensor());
}

}  // namespace sglang
