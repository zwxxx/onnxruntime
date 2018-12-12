// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/element_wise_ops.h"
#include <unsupported/Eigen/SpecialFunctions>

namespace onnxruntime {

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Add,
    7,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Add<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Add,
    7,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Add<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Add,
    7,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    Add<int64_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Sub,
    7,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sub<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Sub,
    7,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Sub<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Sub,
    7,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    Sub<int64_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Mul,
    7,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Mul<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Mul,
    7,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Mul<double>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Mul,
    7,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Mul<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Mul,
    7,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    Mul<int64_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Div,
    7,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Div<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Div,
    7,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Div<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Div,
    7,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    Div<int64_t>);

#define REG_ABS_KERNEL(TYPE)                                                       \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                  \
      Abs,                                                                         \
      6,                                                                           \
      TYPE,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()), \
      Abs<TYPE>);

REG_ABS_KERNEL(float)
REG_ABS_KERNEL(double)
REG_ABS_KERNEL(int8_t)
REG_ABS_KERNEL(int16_t)
REG_ABS_KERNEL(int32_t)
REG_ABS_KERNEL(int64_t)
REG_ABS_KERNEL(uint8_t)
REG_ABS_KERNEL(uint16_t)
REG_ABS_KERNEL(uint32_t)
REG_ABS_KERNEL(uint64_t)

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Neg,
    6,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Neg<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Neg,
    6,
    int8_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()),
    Neg<int8_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Neg,
    6,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Neg<int32_t>);

ONNX_CPU_OPERATOR_KERNEL(
    Floor,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Floor<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Ceil,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Ceil<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Reciprocal,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Reciprocal<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Sqrt,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sqrt<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Pow,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pow<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Exp,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Exp<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Log,
    6,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Log<float>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Sum,
    6, 7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sum_6<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Sum,
    8,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sum_8<float>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Min,
    6, 7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Min_6<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Min,
    8,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Min_8<float>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Max,
    6, 7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Max_6<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Max,
    8,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Max_8<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Not,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    Not);

ONNX_CPU_OPERATOR_KERNEL(
    And,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    And);

ONNX_CPU_OPERATOR_KERNEL(
    Or,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    Or);

ONNX_CPU_OPERATOR_KERNEL(
    Xor,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    Xor);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Less,
    7, 9,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Less<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Less,
    9,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Less<int32_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Greater,
    7, 9,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Greater<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Greater,
    9,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Greater<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Equal,
    7,
    bool,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()),
    Equal<bool>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Equal,
    7,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Equal<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Equal,
    7,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    Equal<int64_t>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Mean,
    6, 7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Mean_6<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Mean,
    8,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Mean_8<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Affine,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Affine<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Scale,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Scale<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Erf,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Erf<float>);

template <typename T>
Status Add<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1) { output = input0 + input1.array(); },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() + input1; },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0 + input1; });
}

template <typename T>
Status Sub<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1) { output = input0 - input1.array(); },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() - input1; },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0 - input1; });
}

template <typename T>
Status Mul<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1) { output = input0 * input1.array(); },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() * input1; },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0.cwiseProduct(input1); });
}

template <typename T>
Status Div<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1) { output = input0 / input1.array(); },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() / input1; },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0.cwiseQuotient(input1); });
}

template <>
Status Floor<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().floor();

  return Status::OK();
}

template <>
Status Ceil<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().ceil();

  return Status::OK();
}

template <>
Status Reciprocal<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).cwiseInverse();

  return Status::OK();
}

template <>
Status Sqrt<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).cwiseSqrt();

  return Status::OK();
}

template <>
Status Pow<float>::Compute(OpKernelContext* context) const {
  const Tensor& Y = *context->Input<Tensor>(1);
  std::function<void(EigenVectorMap<float>, ConstEigenVectorMap<float>, float)> input1scalar =
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) { output = Eigen::pow(input0.array(), input1); };
  if (Y.Shape().Size() == 1) {
    float value = * Y.Data<float>();
    if (value == 2.0) {
      input1scalar = [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float) { output = Eigen::square(input0.array()); };
    }
    else if (value == 3.0) {
      input1scalar = [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float) { output = Eigen::cube(input0.array()); };
    }
  }

  return BroadcastTwo<float, float>(
      *context,
      [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) { output = Eigen::pow(input0, input1.array()); },
      input1scalar,
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) { output = Eigen::pow(input0.array(), input1.array()); });
}

template <>
Status Exp<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().exp();

  return Status::OK();
}

template <>
Status Log<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().log();

  return Status::OK();
}

template <>
Status Sum_6<float>::Compute(OpKernelContext* ctx) const {
  auto input_count = Node().InputArgCount().front();
  ONNXRUNTIME_ENFORCE(input_count >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto sum = EigenMap<float>(*ctx->Output(0, shape));

  if (input_count == 1) {
    sum = EigenMap<float>(data_0);
  } else {
    auto& data_1 = *ctx->Input<Tensor>(1);
    ONNXRUNTIME_ENFORCE(data_1.Shape() == shape, "All inputs must have the same shape");

    sum = EigenMap<float>(data_0) + EigenMap<float>(data_1);
    for (int index = 2; index < input_count; index++) {
      auto& data_n = *ctx->Input<Tensor>(index);
      ONNXRUNTIME_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
      sum += EigenMap<float>(data_n);
    }
  }

  return Status::OK();
}

template <>
Status Sum_8<float>::Compute(OpKernelContext* context) const {
  return BroadcastVariadic<float, float>(
      Node(), *context,
      [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) { output = input0 + input1.array(); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) { output = input0.array() + input1; },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) { output = input0 + input1; });
}

template <>
Status Min_6<float>::Compute(OpKernelContext* ctx) const {
  auto inputCount = Node().InputArgCount().front();
  ONNXRUNTIME_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto min = EigenMap<float>(*ctx->Output(0, shape));

  min = EigenMap<float>(data_0);
  for (int index = 1; index < inputCount; index++) {
    auto& data_n = *ctx->Input<Tensor>(index);
    ONNXRUNTIME_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
    min = min.array().min(EigenMap<float>(data_n).array());
  }

  return Status::OK();
}

template <>
Status Min_8<float>::Compute(OpKernelContext* context) const {
  return BroadcastVariadic<float, float>(
      Node(), *context,
      [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) { output = input1.array().min(input0); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) { output = input0.array().min(input1); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) { output = input0.array().min(input1.array()); });
}

template <>
Status Max_6<float>::Compute(OpKernelContext* ctx) const {
  auto inputCount = Node().InputArgCount().front();
  ONNXRUNTIME_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto max = EigenMap<float>(*ctx->Output(0, shape));

  max = EigenMap<float>(data_0);
  for (int index = 1; index < inputCount; index++) {
    auto& data_n = *ctx->Input<Tensor>(index);
    ONNXRUNTIME_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
    max = max.array().max(EigenMap<float>(data_n).array());
  }

  return Status::OK();
}

template <>
Status Max_8<float>::Compute(OpKernelContext* context) const {
  return BroadcastVariadic<float, float>(
      Node(), *context,
      [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) { output = input1.array().max(input0); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) { output = input0.array().max(input1); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) { output = input0.array().max(input1.array()); });
}

Status Not::Compute(OpKernelContext* context) const {
  auto& input = *context->Input<Tensor>(0);
  auto& output = *context->Output(0, input.Shape());

  EigenMap<bool>(output).array() = !EigenMap<bool>(input).array();
  return Status::OK();
}

Status And::Compute(OpKernelContext* context) const {
  // The scalar cases are special cased, since 'X && true = X' and 'X && false = false'
  return BroadcastTwo<bool, bool>(
      *context,
      [](EigenVectorMap<bool> output, bool input0, ConstEigenVectorMap<bool> input1) {
        if (input0)
          output = input1;
        else
          output.array() = false;
      },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<bool> input0, bool input1) {
        if (input1)
          output = input0;
        else
          output.array() = false;
      },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<bool> input0, ConstEigenVectorMap<bool> input1) { output = input0.array() && input1.array(); });
}

Status Or::Compute(OpKernelContext* context) const {
  // The scalar cases are special cased, since 'X || true = true' and 'X || false = X'
  return BroadcastTwo<bool, bool>(
      *context,
      [](EigenVectorMap<bool> output, bool input0, ConstEigenVectorMap<bool> input1) {
        if (input0)
          output.array() = true;
        else
          output = input1;
      },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<bool> input0, bool input1) {
        if (input1)
          output.array() = true;
        else
          output = input0;
      },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<bool> input0, ConstEigenVectorMap<bool> input1) { output = input0.array() || input1.array(); });
}

Status Xor::Compute(OpKernelContext* context) const {
  // The scalar cases are special cased, since 'X ^ true = !X' and 'X ^ false = X'
  return BroadcastTwo<bool, bool>(
      *context,
      [](EigenVectorMap<bool> output, bool input0, ConstEigenVectorMap<bool> input1) {
        if (input0)
          output.array() = !input1.array();
        else
          output = input1;
      },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<bool> input0, bool input1) {
        if (input1)
          output.array() = !input0.array();
        else
          output = input0;
      },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<bool> input0, ConstEigenVectorMap<bool> input1) { output = input0.array() ^ input1.array(); });
}

template <typename T>
Status Equal<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, bool>(
      *context,
      [](EigenVectorMap<bool> output, T input0, ConstEigenVectorMap<T> input1) { output = input1.array() == input0; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() == input1; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0.array() == input1.array(); });
}

template <typename T>
Status Less<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, bool>(
      *context,
      [](EigenVectorMap<bool> output, T input0, ConstEigenVectorMap<T> input1) { output = input1.array() > input0; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() < input1; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0.array() < input1.array(); });
}

template <typename T>
Status Greater<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, bool>(
      *context,
      [](EigenVectorMap<bool> output, T input0, ConstEigenVectorMap<T> input1) { output = input1.array() < input0; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() > input1; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0.array() > input1.array(); });
}

template <>
Status Mean_6<float>::Compute(OpKernelContext* ctx) const {
  auto inputCount = Node().InputArgCount().front();
  ONNXRUNTIME_ENFORCE(inputCount >= 1, "Must have 1 or more inputs");
  auto& data_0 = *ctx->Input<Tensor>(0);
  auto& shape = data_0.Shape();
  auto mean = EigenMap<float>(*ctx->Output(0, shape));

  if (inputCount == 1) {
    mean = EigenMap<float>(data_0);
  } else {
    auto& data_1 = *ctx->Input<Tensor>(1);
    ONNXRUNTIME_ENFORCE(data_1.Shape() == shape, "All inputs must have the same shape");

    mean = EigenMap<float>(data_0) + EigenMap<float>(data_1);
    for (int index = 2; index < inputCount; index++) {
      auto& data_n = *ctx->Input<Tensor>(index);
      ONNXRUNTIME_ENFORCE(data_n.Shape() == shape, "All inputs must have the same shape");
      mean += EigenMap<float>(data_n);
    }
  }

  // Take the mean
  float weight = 1.0f / static_cast<float>(inputCount);
  mean = mean * weight;

  return Status::OK();
}

template <>
Status Mean_8<float>::Compute(OpKernelContext* context) const {
  // Do a sum exactly the same as in Sum_8
  Status status = BroadcastVariadic<float, float>(
      Node(), *context,
      [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) { output = input0 + input1.array(); },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) { output = input0.array() + input1; },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) { output = input0 + input1; });
  if (!status.IsOK())
    return status;

  // Now divide by the input count to get the mean
  EigenMap<float>(*context->Output<Tensor>(0)) *= 1.0f / static_cast<float>(Node().InputArgCount().front());
  return Status::OK();
}

template <>
Status Affine<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());
  MakeEigenArrayMap<float>(Y) = alpha_ * MakeEigenArrayMap<float>(X) + beta_;
  return Status::OK();
}

template <typename T>
class Sin final : public OpKernel {
 public:
  Sin(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).sin();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Sin,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Sin<float>);

template <typename T>
class Cos final : public OpKernel {
 public:
  Cos(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).cos();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Cos,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Cos<float>);

template <typename T>
class Tan final : public OpKernel {
 public:
  Tan(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).tan();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Tan,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Tan<float>);

template <typename T>
class Asin final : public OpKernel {
 public:
  Asin(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).asin();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Asin,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Asin<float>);

template <typename T>
class Acos final : public OpKernel {
 public:
  Acos(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).acos();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Acos,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Acos<float>);

template <typename T>
class Atan final : public OpKernel {
 public:
  Atan(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& X = *context->Input<Tensor>(0);
    auto& Y = *context->Output(0, X.Shape());
    MakeEigenArrayMap<float>(Y) = MakeEigenArrayMap<float>(X).atan();
    return Status::OK();
  }
};

ONNX_CPU_OPERATOR_KERNEL(
    Atan,
    7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Atan<float>);

template <>
Status PRelu<float>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<float, float>(
      *context,
      [](EigenVectorMap<float> output, float input0, ConstEigenVectorMap<float> input1) {
        if (input0 > 0)
          output.array() = input0;
        else
          output = input0 * input1.array();
      },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, float input1) {
        output = (input0.array() > 0).select(input0, input0 * input1);
      },
      [](EigenVectorMap<float> output, ConstEigenVectorMap<float> input0, ConstEigenVectorMap<float> input1) {
        output = (input0.array() > 0).select(input0, input0.cwiseProduct(input1));
      });
}

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    PRelu,
    7,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    PRelu<float>);

// This is a special case version of TBroadcaster just for Expand that only has a shape as the second parameter
template <typename T>
struct TBroadcasterExpand {
  TBroadcasterExpand(const Tensor& input, const std::vector<int64_t>& shape)
      : input_tensor_(input),
        broadcaster_(input.Shape().GetDims(), shape) {
  }

  TensorShape GetOutputShape() const { return TensorShape(broadcaster_.output_shape_); }
  size_t GetSpanSize() const { return span_size_; }

  bool IsInput0Scalar() const { return broadcaster_.iterator1_.deltas_.front() == 0; }

  T NextScalar() { return *Next(); }

  ConstEigenVectorMap<T> NextEigen() { return ConstEigenVectorMap<T>(Next(), span_size_); }

 private:
  const T* Next() { return input_ + broadcaster_.iterator1_.AdvanceBy(span_size_); }

  const Tensor& input_tensor_;
  Broadcaster broadcaster_;
  size_t span_size_{broadcaster_.GetSpanSize()};

  const T* input_{input_tensor_.template Data<T>()};
};

template <typename T>
Status Expand_8<T>::Compute(OpKernelContext* context) const {
  auto& tensor_shape = *context->Input<Tensor>(1);
  ONNXRUNTIME_ENFORCE(tensor_shape.Shape().NumDimensions() == 1, "Shape must be 1 dimensional as it's tensor data is a shape");

  // Turn the shape tensor data into an actual shape
  const int64_t* p_shape = tensor_shape.template Data<int64_t>();
  std::vector<int64_t> shape{p_shape, p_shape + tensor_shape.Shape().Size()};

  TBroadcasterExpand<T> bc(*context->Input<Tensor>(0), shape);
  TBroadcastOutput<T> output(bc.GetSpanSize(), *context->Output(0, bc.GetOutputShape()));

  // This doesn't use BroadcastLoop since there is no second tensor, just duplicating the first
  if (bc.IsInput0Scalar()) {
    // Input0 being a scalar is the only special case here, since we're duplicating a single value
    while (output)
      output.NextEigenOutput().array() = bc.NextScalar();
  } else {
    // Input1 being a scalar doesn't matter (as there's no actual input1). We're still duplicating Input0 in the same sized chunks
    while (output)
      output.NextEigenOutput() = bc.NextEigen();
  }
  return Status::OK();
}

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Expand,
    8,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Expand_8<float>);

template <>
Status Scale<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());
  EigenMap<float>(Y) = scale_ * EigenMap<float>(X);
  return Status::OK();
}

template <>
Status Erf<float>::Compute(OpKernelContext* context) const {
  auto X_ptr = context->Input<Tensor>(0);
  ONNXRUNTIME_ENFORCE(X_ptr != nullptr);
  auto& X = *X_ptr;
  auto& Y = *context->Output(0, X.Shape());

  EigenMap<float>(Y) = EigenMap<float>(X).array().erf();

  return Status::OK();
}

}  // namespace onnxruntime
