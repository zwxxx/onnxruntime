#include "reverse_sequence.h"
#include "onnx/defs/schema.h"

namespace onnxruntime {
namespace contrib {

Status ReverseSequence::Compute(OpKernelContext* ctx) const {
  (void)ctx;
  return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,  
                                 "Not implemented yet");
}

/* Range operator */
ONNX_OPERATOR_KERNEL_EX(
    ReverseSequence,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", {
            DataTypeImpl::GetTensorType<float>(), 
            DataTypeImpl::GetTensorType<double>(),
            DataTypeImpl::GetTensorType<int16_t>(), 
            DataTypeImpl::GetTensorType<int32_t>(), 
            DataTypeImpl::GetTensorType<int64_t>()})
        .TypeConstraint("TIndex", {
            DataTypeImpl::GetTensorType<int32_t>(), 
            DataTypeImpl::GetTensorType<int64_t>()}),
    ReverseSequence);

}  // namespace contrib
}  // namespace onnxruntime
