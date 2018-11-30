// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

class ReverseSequence : public OpKernel {
public:
  explicit ReverseSequence(const OpKernelInfo& info) : OpKernel(info) {
      ONNXRUNTIME_ENFORCE(info.GetAttr<int64_t>("seq_axis", &seq_axis_).IsOK());
      if (!info.GetAttr<int64_t>("batch_axis", &batch_axis_).IsOK()) {
          batch_axis_ = 0;
      }
      ONNXRUNTIME_ENFORCE(seq_axis_ >= 0 && batch_axis_ >= 0 && seq_axis_ != batch_axis_,
                          "seq_axis and batch axis should all greater or equal than 0 and different with each other",
                          "seq_axis=", seq_axis_, ", batch_axis=", batch_axis_);
  }

  Status Compute(OpKernelContext* context) const override;

private:
  int64_t seq_axis_;
  int64_t batch_axis_;

  template <typename TData, typename TIndex>
  Status ComputeImpl(OpKernelContext* context) const;
};

}  // namespace contrib
}  // namespace onnxruntime
