// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/constant_fill.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    ConstantFill,
    1,
    KernelDefBuilder().TypeConstraint("T1",
                                      std::vector<MLDataType>{
                                          DataTypeImpl::GetTensorType<float>(),
                                          DataTypeImpl::GetTensorType<int32_t>(),
                                          DataTypeImpl::GetTensorType<int64_t>(),
                                          DataTypeImpl::GetTensorType<bool>(),
                                      })
                      .TypeConstraint("T2",
                                      std::vector<MLDataType>{
                                          DataTypeImpl::GetTensorType<float>(),
                                          DataTypeImpl::GetTensorType<int32_t>(),
                                          DataTypeImpl::GetTensorType<int64_t>(),
                                          DataTypeImpl::GetTensorType<bool>(),
                                      }),
    ConstantFill);

Status ConstantFill::Compute(OpKernelContext* context) const {
  switch (dtype_) {
    case onnx::TensorProto_DataType_FLOAT:
      return ComputeImpl<float>(context);
    case onnx::TensorProto_DataType_INT32:
      return ComputeImpl<int32_t>(context);
    case onnx::TensorProto_DataType_INT64:
      return ComputeImpl<int64_t>(context);
    case onnx::TensorProto_DataType_BOOL:
      return ComputeImpl<bool>(context);
    default:
      ONNXRUNTIME_THROW("Unexpected 'dtype' value: ", dtype_);
  }
}

template <typename T>
Status ConstantFill::ComputeImpl(OpKernelContext* context) const {
  TensorShape shape;
  if (input_as_shape_) {
    auto* t1 = context->Input<Tensor>(0);
    std::vector<int64_t> dims = t1->Shape().GetDims();
    dims.insert(dims.end(), extra_shape_.begin(), extra_shape_.end());
    shape = TensorShape(dims);
  } else {
    shape = TensorShape(shape_);
  }

  auto value = static_cast<T>(value_);
  auto* t2 = context->Output(0, shape);
  auto* data = t2->MutableData<T>();
  for (int64_t i = 0; i < shape.Size(); ++i) {
    data[i] = value;
  }

  return Status::OK();
}  // namespace onnxruntime

}  // namespace onnxruntime
