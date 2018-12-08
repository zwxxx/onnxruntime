using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics.Tensors;

namespace Microsoft.ML.OnnxRuntime
{
    public static class Utils
    {
        /// <summary>
        /// Parses a Tensor<typeparamref name="T"/> object from a serialized protobuf representation of Onnx Tensor 
        /// </summary>
        /// <typeparam name="T">Element type of the Tensor</typeparam>
        /// <param name="serializedBuffer"></param>
        /// <returns>The parsed Tensor</returns>
        public static Tensor<T> ParseTensorFromProto<T>(byte[] serializedBuffer)
        {
            IntPtr onnxValue = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(
                NativeMethods.ONNXRuntimeTensorProtoToONNXValue(
                    NativeMemoryAllocator.DefaultInstance.Handle,
                    serializedBuffer,
                    serializedBuffer.Length,
                    out onnxValue
                ));

            NativeOnnxTensorMemory<T> nativeTensorWrapper = new NativeOnnxTensorMemory<T>(onnxValue);
            DenseTensor<T> result = new DenseTensor<T>(nativeTensorWrapper.Memory, nativeTensorWrapper.Dimensions);

            return result;
        }
    }
}
