#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "sglang/server/args.h"

namespace sglang {
namespace {

TEST(ServerArgsTest, ParseBasicFlags) {
  auto args = ServerArgsParser::parse(std::vector<std::string>{
      "--model-path",
      "/tmp/model",
      "--dtype",
      "float16",
      "--host",
      "0.0.0.0",
      "--port",
      "8080",
      "--num-tokenizer-threads",
      "2",
      "--max-running-requests",
      "8",
      "--max-prefill-length",
      "64",
      "--memory-ratio",
      "0.25",
      "--page-size",
      "1",
      "--num-pages",
      "256",
      "--max-seq-len-override",
      "1024",
      "--dummy-weight",
      "--cache-type",
      "radix",
      "--enable-overlap-scheduling",
  });

  EXPECT_EQ(args.model_path, "/tmp/model");
  EXPECT_EQ(args.dtype, torch::kFloat16);
  EXPECT_EQ(args.server_host, "0.0.0.0");
  EXPECT_EQ(args.server_port, 8080);
  EXPECT_EQ(args.num_tokenizer_threads, 2);
  EXPECT_EQ(args.max_running_req, 8);
  EXPECT_EQ(args.max_extend_tokens, 64);
  EXPECT_FLOAT_EQ(args.memory_ratio, 0.25F);
  EXPECT_EQ(args.page_size, 1);
  ASSERT_TRUE(args.num_pages_override.has_value());
  EXPECT_EQ(*args.num_pages_override, 256);
  ASSERT_TRUE(args.max_seq_len_override.has_value());
  EXPECT_EQ(*args.max_seq_len_override, 1024);
  EXPECT_TRUE(args.use_dummy_weight);
  EXPECT_EQ(args.cache_type, "radix");
  EXPECT_TRUE(args.enable_overlap_scheduling);
}

TEST(ServerArgsTest, OverlapSchedulingDefaultsOnAndCanBeDisabled) {
  auto default_args = ServerArgsParser::parse(std::vector<std::string>{
      "--model-path",
      "/tmp/model",
  });
  EXPECT_TRUE(default_args.enable_overlap_scheduling);

  auto disabled_args = ServerArgsParser::parse(std::vector<std::string>{
      "--model-path",
      "/tmp/model",
      "--disable-overlap-scheduling",
  });
  EXPECT_FALSE(disabled_args.enable_overlap_scheduling);

  auto reenabled_args = ServerArgsParser::parse(std::vector<std::string>{
      "--model-path",
      "/tmp/model",
      "--disable-overlap-scheduling",
      "--enable-overlap-scheduling",
  });
  EXPECT_TRUE(reenabled_args.enable_overlap_scheduling);
}

TEST(ServerArgsTest, ShellModeTightensDefaults) {
  auto args = ServerArgsParser::parse(std::vector<std::string>{
      "--model-path",
      "/tmp/model",
      "--shell-mode",
      "--max-running-requests",
      "8",
  });

  EXPECT_TRUE(args.shell_mode);
  EXPECT_TRUE(args.silent_output);
  EXPECT_EQ(args.max_running_req, 1);
  EXPECT_TRUE(args.enable_overlap_scheduling);
}

TEST(ServerArgsTest, CudaGraphFlagAndDisableFlagWork) {
  auto default_args = ServerArgsParser::parse(std::vector<std::string>{
      "--model-path",
      "/tmp/model",
  });
  EXPECT_FALSE(default_args.enable_cuda_graph);

  auto enabled_args = ServerArgsParser::parse(std::vector<std::string>{
      "--model-path",
      "/tmp/model",
      "--graph",
      "16",
      "--cuda-graph-batch-sizes",
      "1,2,4,8,16",
  });
  EXPECT_TRUE(enabled_args.enable_cuda_graph);
  ASSERT_TRUE(enabled_args.cuda_graph_max_batch_size.has_value());
  EXPECT_EQ(*enabled_args.cuda_graph_max_batch_size, 16);
  EXPECT_EQ(enabled_args.cuda_graph_batch_sizes, (std::vector<int>{1, 2, 4, 8, 16}));

  auto disabled_args = ServerArgsParser::parse(std::vector<std::string>{
      "--model-path",
      "/tmp/model",
      "--disable-graph",
  });
  EXPECT_FALSE(disabled_args.enable_cuda_graph);
}

TEST(ServerArgsTest, ToSchedulerConfigMapsFields) {
  ServerArgs args;
  args.model_path = "/tmp/model";
  args.dtype = torch::kBFloat16;
  args.max_running_req = 6;
  args.page_size = 1;
  args.memory_ratio = 0.3F;
  args.use_dummy_weight = true;
  args.enable_cuda_graph = true;
  args.cuda_graph_max_batch_size = 16;
  args.cuda_graph_batch_sizes = {1, 2, 4, 8, 16};
  args.max_seq_len_override = 2048;
  args.num_pages_override = 128;
  args.max_extend_tokens = 96;
  args.cache_type = "radix";
  args.enable_overlap_scheduling = true;

  auto config = args.to_scheduler_config();
  EXPECT_EQ(config.model_path, args.model_path);
  EXPECT_EQ(config.dtype, args.dtype);
  EXPECT_EQ(config.max_running_req, args.max_running_req);
  EXPECT_EQ(config.page_size, args.page_size);
  EXPECT_FLOAT_EQ(config.memory_ratio, args.memory_ratio);
  EXPECT_TRUE(config.use_dummy_weight);
  EXPECT_TRUE(config.enable_cuda_graph);
  ASSERT_TRUE(config.cuda_graph_max_batch_size.has_value());
  EXPECT_EQ(*config.cuda_graph_max_batch_size, 16);
  EXPECT_EQ(config.cuda_graph_batch_sizes, (std::vector<int>{1, 2, 4, 8, 16}));
  ASSERT_TRUE(config.max_seq_len_override.has_value());
  EXPECT_EQ(*config.max_seq_len_override, 2048);
  ASSERT_TRUE(config.num_pages_override.has_value());
  EXPECT_EQ(*config.num_pages_override, 128);
  EXPECT_EQ(config.max_extend_tokens, 96);
  EXPECT_EQ(config.cache_type, "radix");
  EXPECT_TRUE(config.enable_overlap_scheduling);
}

}  // namespace
}  // namespace sglang
