#include "reverse_sequence.h"
#include "onnx/defs/schema.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

namespace contrib {

Status ReverseSequence::Compute(OpKernelContext* ctx) const {
  auto data_type = (*ctx->Input<Tensor>(0)).DataType();
  auto index_type = (*ctx->Input<Tensor>(1)).DataType();
  bool is_index_64 = index_type == DataTypeImpl::GetType<int64_t>();

  if (data_type == DataTypeImpl::GetType<float>()) {
    return is_index_64 ? ComputeImpl<float, int64_t>(ctx) : ComputeImpl<float, int32_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<int32_t>()) {
    return is_index_64 ? ComputeImpl<int32_t, int64_t>(ctx) : ComputeImpl<int32_t, int32_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<int16_t>()) {
    return is_index_64 ? ComputeImpl<int16_t, int64_t>(ctx) : ComputeImpl<int16_t, int32_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<int64_t>()) {
    return is_index_64 ? ComputeImpl<int64_t, int64_t>(ctx) : ComputeImpl<int64_t, int32_t>(ctx);
  }
  else  if (data_type == DataTypeImpl::GetType<double>()) {
    return is_index_64 ? ComputeImpl<double, int64_t>(ctx) : ComputeImpl<double, int32_t>(ctx);
  }

  return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,  "Not implemented yet");
}


template <typename TData, typename TIndex>
Status ReverseSequence::ComputeImpl(OpKernelContext* ctx) const {
  const Tensor& input = *(ctx->Input<Tensor>(0));
  const std::vector<int64_t>& dims = input.Shape().GetDims();
  int64_t num_dims = input.Shape().NumDimensions();
  
  ONNXRUNTIME_ENFORCE(seq_axis_ < num_dims, "Input number of dims:", num_dims, " should greater than seq_axis:", seq_axis_);
  ONNXRUNTIME_ENFORCE(batch_axis_ < num_dims, "Input number of dims:", num_dims, " should greater than batch_axis_:", batch_axis_);
  
  int64_t batch_size = dims[batch_axis_];
  int64_t seq_size = dims[seq_axis_];

  const Tensor& seq_lengths = *(ctx->Input<Tensor>(1));
  int64_t seq_lengths_ndims = seq_lengths.Shape().NumDimensions();  
  ONNXRUNTIME_ENFORCE(seq_lengths_ndims == 0 || seq_lengths_ndims == 1, 
                     "seq_lengths must be 0-D or 1-D tensor, yet found:", seq_lengths_ndims);

  int64_t seq_lengths_size = seq_lengths.Shape().Size();
  ONNXRUNTIME_ENFORCE(seq_lengths_size == batch_size || seq_lengths_size == 1, 
                     "Wrong seq_lengths Size:", seq_lengths_size, " expect 1 or ", batch_size);
  const TIndex* seq_lengths_value = seq_lengths.template Data<TIndex>();

  for (int64_t i = 0; i < seq_lengths_size; ++i) {
      ONNXRUNTIME_ENFORCE(seq_lengths_value[i] > 0 && seq_lengths_value[i] <= seq_size,
                          "Each seq_len should > 0 and <= seq_size:", seq_size, ", but found:", seq_lengths_value[i]);
  }

  //reshape dims like: [h, B/S, m, S/B, t]
  int64_t s[3] = {1LL, 1LL, 1LL}; // h, m, t
  int meet = 0;  // number of times meet Batch_axis or Seq_axis
  int64_t merged_batch_size = batch_size;
  for (int64_t i = 0; i < num_dims; ++i) {
      if (i == batch_axis_) {
          merged_batch_size *= s[meet++];
      }
      else if (i == seq_axis_) {
          meet++;
      }
      else {
          s[meet] *= dims[i];
      }
  }

  int64_t width = s[2];
  int64_t pre_seq_size = ((batch_axis_ < seq_axis_) ? s[1] : s[0]); // merged dimension right before sequence
  int64_t merged_batch_strides = s[2], seq_strides = s[2], pre_seq_strides;
  if (batch_axis_ < seq_axis_) { // [(h B) m S t]
    pre_seq_strides = seq_strides * seq_size;
    merged_batch_strides = pre_seq_strides * pre_seq_size;
  }
  else {  // [h S (m B) t]
    seq_strides = merged_batch_strides * merged_batch_size;
    pre_seq_strides = seq_strides * seq_size;
  }

  TData* y = ctx->Output(0, input.Shape())->template MutableData<TData>();
  const TData* x = input.template Data<TData>();

  #pragma omp parallel for
  for (int64_t merged_batch = 0; merged_batch < merged_batch_size; ++merged_batch) {
    int64_t batch = merged_batch % batch_size;
    int64_t seq_len = int64_t{seq_lengths_value[batch % seq_lengths_size]};
    int64_t mb_pos = merged_batch * merged_batch_strides;
    for (int64_t pre = 0; pre < pre_seq_size; ++pre) {
      int64_t pre_seq_pos = mb_pos + pre * pre_seq_strides;
      if (batch_axis_ < seq_axis_) {
        // seq_axis, and following data are consequtive, so using matrix level reverse
        EigenMatrixMapRowMajor<TData> out_reverse(y + pre_seq_pos, seq_len, width);
        out_reverse = ConstEigenMatrixMapRowMajor<TData>(x + pre_seq_pos, seq_len, width).colwise().reverse();

        EigenMatrixMapRowMajor<TData> out_copy(y + pre_seq_pos + seq_strides * seq_len, seq_size - seq_len, width);
        out_copy = ConstEigenMatrixMapRowMajor<TData>(x + pre_seq_pos + seq_strides * seq_len, seq_size - seq_len, width).colwise().reverse();
      }
      else {
        int64_t offset = pre_seq_pos;
        // reverse the [seq_len, width]
        for (int64_t seq = 0; seq < seq_len; ++seq) {
          EigenVectorMap<TData> out_vec(y + pre_seq_pos + seq_strides * (seq_len - seq - 1), width);
          out_vec = ConstEigenVectorMap<TData>(x + offset, width);
          offset += seq_strides;
        }

        for (int64_t seq = seq_len; seq < seq_size; ++seq) {
          EigenVectorMap<TData> out_vec(y + offset, width);
          out_vec = ConstEigenVectorMap<TData>(x + offset, width);
          offset += seq_strides;
        }
      }
    }
  }

  return Status::OK();
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
