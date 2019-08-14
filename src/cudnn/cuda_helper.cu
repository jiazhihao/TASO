#include "xflow/cuda_helper.h"
using namespace XFlow;

__global__
void assign_kernel(float* ptr, int size, float value)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = value;
  }
}

cudnnActivationMode_t get_activation_mode(ActiMode activation)
{
  switch (activation) {
    case AC_MODE_SIGMOID:
      return CUDNN_ACTIVATION_SIGMOID;
    case AC_MODE_RELU:
      return CUDNN_ACTIVATION_RELU;
    case AC_MODE_TANH:
      return CUDNN_ACTIVATION_TANH;
    default:
      assert(false);
  }
  // return RELU as default
  return CUDNN_ACTIVATION_RELU;
}

void helperSetTensorDescriptor(const Tensor& tensor,
                               cudnnTensorDescriptor_t tensorDesc)
{
  switch(tensor.numDim) {
    case 1:
    {
      int dims[] = {tensor.dim[0], 1, 1};
      int strides[] = {tensor.stride[0], 1, 1};
      checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc, CUDNN_DATA_FLOAT,
                                            3, dims, strides));
      break;
    }
    case 2:
    {
      int dims[] = {tensor.dim[0], tensor.dim[1], 1, 1};
      int strides[] = {tensor.stride[0], tensor.stride[1], 1, 1};
      checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc, CUDNN_DATA_FLOAT,
                                            4, dims, strides));
      break;
    }
    default:
    {
      assert(tensor.numDim >= 3);
      checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc, CUDNN_DATA_FLOAT,
          tensor.numDim, tensor.dim, tensor.stride));
    }
  }
}

