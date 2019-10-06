#ifndef _CUDA_HELPER_H_
#define _CUDA_HELPER_H_

#include <sstream>
#include <iostream>
#include "taso/ops.h"
#include <cudnn.h>

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCUDA(status) do {                                         \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int BLOCK_SIZE_LIMIT = 32768;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  int ret = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  return (ret > BLOCK_SIZE_LIMIT) ? BLOCK_SIZE_LIMIT : ret;
}

void helperSetTensorDescriptor(const taso::Tensor& tensor,
                               cudnnTensorDescriptor_t tensorDesc);

void helperSetBroadcastableTensorDescriptor(const taso::Tensor& input,
                                            const taso::Tensor& output,
                                            cudnnTensorDescriptor_t tensorDesc);

__global__
void assign_kernel(float* ptr, int size, float value);

__global__
void copy_kernel(float* dst, const float* src, int size);

cudnnActivationMode_t get_activation_mode(taso::ActiMode activation);
#endif
