// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(CpuReverseSequenceTest, BatchSequenceX4_int) {
  OpTester test("ReverseSequence", 1, onnxruntime::kMSDomain);
  int32_t batch_size = 4;
  int32_t max_seq_len = 5;
  int32_t last_dim_size = 2;
  std::vector<int32_t> seq_lengths = {1, 3, 5, 4};
  std::vector<int32_t> input = {  // [batch_size, max_seq_len, last_dim_size]
      111,112,  0,  0,  0,  0,  0,  0,  0,  0,
      211,212,221,222,231,232,  0,  0,  0,  0,
      311,312,321,322,331,332,341,342,351,352,
      411,412,421,422,431,432,441,442,  0,  0
  };
  int64_t batch_axis = 0;
  int64_t seq_axis = 1;

  std::vector<int32_t> expected_output = {  // [batch_size, max_seq_len, last_dim_size]
      111,112,  0,  0,  0,  0,  0,  0,  0,  0,
      231,232,221,222,211,212,  0,  0,  0,  0,
      351,352,341,342,331,332,321,322,311,312,
      441,442,431,432,421,422,411,412,  0,  0
  };

  test.AddAttribute<int64_t>("batch_axis", batch_axis);
  test.AddAttribute<int64_t>("seq_axis", seq_axis);
  test.AddInput<int32_t>("input", {batch_size, max_seq_len, last_dim_size}, input);
  test.AddInput<int32_t>("seq_lengths", {batch_size}, seq_lengths);
  test.AddOutput<int32_t>("Y", {batch_size, max_seq_len, last_dim_size}, expected_output);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime