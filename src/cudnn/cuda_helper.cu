#include "taso/cuda_helper.h"
using namespace taso;

__global__
void assign_kernel(float* ptr, int size, float value)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = value;
  }
}

__global__
void copy_kernel(float* dst, const float* src, int size)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    dst[i] = src[i];
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
      int dims[] = {tensor.dim[0], 1, 1, 1};
      int strides[] = {tensor.stride[0], 1, 1, 1};
      checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc, CUDNN_DATA_FLOAT,
                                            4, dims, strides));
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

void helperSetBroadcastableTensorDescriptor(const Tensor& input,
                                            const Tensor& output,
                                            cudnnTensorDescriptor_t tensorDesc)
{
  int dims[16], strides[16];
  assert(output.numDim <= 16);
  assert(input.numDim <= output.numDim);
  assert(input.default_layout());
  assert(output.default_layout());
  for (int i = 0; i < output.numDim; i++) {
    if (i < input.numDim) {
      strides[output.numDim-1-i] = input.stride[input.numDim-1-i];
      dims[output.numDim-1-i] = input.dim[input.numDim-1-i];
    } else {
      strides[output.numDim-1-i] = input.stride[0];
      dims[output.numDim-1-i] = 1;
    }
  }
  int num_dim = output.numDim;
  if (num_dim < 4) {
    num_dim = 4;
    for (int i = output.numDim; i < num_dim; i++) {
      dims[i] = 1;
      strides[i] = 1;
    }
  }
  //for (int i = 0; i < num_dim; i++)
  //  printf("dims[%d] = %d input.dim(%d) output.dim(%d)\n", i, dims[i], input.dim[i], output.dim[i]);
  //for (int i = 0; i < num_dim; i++)
  //  printf("strides[%d] = %d input.stride(%d) output.stride(%d)\n", i, strides[i], input.stride[i], output.stride[i]);
 
  checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc, CUDNN_DATA_FLOAT,
      num_dim, dims, strides));
}

