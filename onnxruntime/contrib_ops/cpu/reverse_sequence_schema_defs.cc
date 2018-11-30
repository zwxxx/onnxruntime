// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse_sequence_schema_defs.h"

#include "core/graph/constants.h"
#include "core/graph/op.h"

namespace onnxruntime {
namespace contrib {

using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::OPTIONAL;
using ::ONNX_NAMESPACE::OpSchema;

// This Doc based on LSTM_ver7, and modification
static const char* ReverseSequence_ver1_doc = R"DOC(
Reverses variable length slices. Generally used in the RNN backward phrase.
Attrs:
  seq_axis: INT. specify the seq axis. max_seq_len = input.dims[seq_axis]
  batch_axis: INT. specify the batch axis, default 0. batch_size = input.dims[batch_axis]
Input:
  input: Tensor to reverse. Normally it is of shape [batch_size, max_seq_len, ...] or [max_seq_len, batch_size, ...]
  seq_lengths: dtype int32/int64. Either a scalar or 1D [batch_size]. All of its elements <= max_seq_len.
)DOC";

OpSchema& RegisterReverseSequenceOpSchema(OpSchema&& op_schema){
  return op_schema
    .SetDomain(kMSDomain)
    .SinceVersion(1)
    .TypeConstraint(
        "T",
        {"tensor(float)", "tensor(double)", "tensor(int16)", "tensor(int32)", "tensor(int64)" },
        "Constrain input and output types.")
    .TypeConstraint(
        "TIndex",
        {"tensor(int32)", "tensor(int64)" },
        "Index type.")
    .Attr(
        "seq_axis",
        "axis of the sequence.",
        AttributeProto::INT)
    .Attr(
        "batch_axis",
        "axis of the batch. default 0.",
        AttributeProto::INT,
        OPTIONAL)
    .Input(
        0,
        "input",
        "Tensor to reverse",
        "T")
    .Input(
        1,
        "seq_lengths",
        "Either scalar of int32/int64, or 1D Tensor of [batch_size].",
        "TIndex")
    .Output(
        0,
        "Y",
        "Reversed result.",
        "T")
    .SetDoc(ReverseSequence_ver1_doc);
}

}  // namespace contrib
}  // namespace onnxruntime
