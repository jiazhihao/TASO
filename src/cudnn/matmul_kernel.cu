/* Copyright 2019 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "taso/ops.h"
#include "taso/cuda_helper.h"
using namespace taso;

void Matmul::map(void)
{
  // create descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  helperSetTensorDescriptor(outputs[0], outputTensor);
  if (activation != AC_MODE_NONE) {
    cudnnActivationMode_t mode;
    switch (activation) {
      case AC_MODE_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case AC_MODE_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case AC_MODE_TANH:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      default:
        assert(false);
    }
    checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    checkCUDNN(cudnnSetActivationDescriptor(actiDesc, mode,
        CUDNN_NOT_PROPAGATE_NAN, 0.0));
  }
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Matmul::unmap(void)
{
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  if (activation != AC_MODE_NONE) {
    checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  }
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Matmul::forward(bool block)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int numDim = outputs[0].numDim;
  int m = inputs[0].dim[numDim-2];
  int n = inputs[1].dim[numDim-1];
  int k = inputs[0].dim[numDim-1];
  cublasOperation_t transA, transB;
  int lda, ldb, ldc;
  if (inputs[0].stride[numDim-2] == 1) {
    transA = CUBLAS_OP_N;
    lda = inputs[0].stride[numDim-1];
  } else {
    assert(inputs[0].stride[numDim-1] == 1);
    transA = CUBLAS_OP_T;
    lda = inputs[0].stride[numDim-2];
  }
  if (inputs[1].stride[numDim-2] == 1) {
    transB = CUBLAS_OP_N;
    ldb = inputs[1].stride[numDim-1];
  } else {
    assert(inputs[1].stride[numDim-1] == 1);
    transB = CUBLAS_OP_T;
    ldb = inputs[1].stride[numDim-2];
  }
  ldc = outputs[0].stride[numDim-1];
  if (numDim == 2) {
    // Normal 2D Matmul
    checkCUDA(cublasSgemm(model->blas, transA, transB,
        m, n, k, &alpha, (float*)inputs[0].data_ptr, lda,
        (float*)inputs[1].data_ptr, ldb, &beta, (float*)outputs[0].data_ptr, ldc));
  } else {
    // Batched Matmul
    int strideA = inputs[0].stride[numDim-3];
    int strideB = inputs[1].stride[numDim-3];
    int strideC = outputs[0].stride[numDim-3];
    int batch = 1;
    for (int i = 0; i < numDim-2; i++)
      batch *= outputs[0].dim[i];
    checkCUDA(cublasSgemmStridedBatched(model->blas, transA, transB,
        m, n, k, &alpha, (float*)inputs[0].data_ptr, lda, strideA,
        (float*)inputs[1].data_ptr, ldb, strideB,
        &beta, (float*)outputs[0].data_ptr, ldc, strideC, batch));
  }
  if (activation != AC_MODE_NONE)
    checkCUDNN(cudnnActivationForward(model->dnn, actiDesc,
        &alpha, outputTensor, outputs[0].data_ptr,
        &beta, outputTensor, outputs[0].data_ptr));
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Matmul::set_layout(void)
{
  // CuBLAS uses column-major.
  int numDim = outputs[0].numDim;
  outputs[0].stride[numDim-2] = 1;
  outputs[0].stride[numDim-1] = outputs[0].dim[numDim-2];
  int size = outputs[0].dim[numDim-2] * outputs[0].dim[numDim-1];
  for (int i = numDim-3; i >= 0; i--) {
    outputs[0].stride[i] = size;
    size *= outputs[0].dim[i];
  }
  assert(size == outputs[0].volume());
}

void Model::measure_matmul_cost(Matmul* mm)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int numDim = mm->outputs[0].numDim;
  int m = mm->inputs[0].dim[numDim-2];
  int n = mm->inputs[1].dim[numDim-1];
  int k = mm->inputs[0].dim[numDim-1];
  cublasOperation_t transA, transB;
  int lda, ldb, ldc;
  if (mm->inputs[0].stride[numDim-2] == 1) {
    transA = CUBLAS_OP_N;
    lda = mm->inputs[0].stride[numDim-1];
  } else {
    assert(mm->inputs[0].stride[numDim-1] == 1);
    transA = CUBLAS_OP_T;
    lda = mm->inputs[0].stride[numDim-2];
  }
  if (mm->inputs[1].stride[numDim-2] == 1) {
    transB = CUBLAS_OP_N;
    ldb = mm->inputs[1].stride[numDim-1];
  } else {
    assert(mm->inputs[1].stride[numDim-1] == 1);
    transB = CUBLAS_OP_T;
    ldb = mm->inputs[1].stride[numDim-2];
  }
  ldc = mm->outputs[0].stride[numDim-1];

  if (mm->activation != AC_MODE_NONE) {
    cudnnActivationMode_t mode;
    switch (mm->activation) {
      case AC_MODE_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case AC_MODE_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case AC_MODE_TANH:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      default:
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(actiDesc, mode,
        CUDNN_NOT_PROPAGATE_NAN, 0.0));
  }
  helperSetTensorDescriptor(mm->outputs[0], outputTensor);

  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < WARMUP_TIMES + REPEAT_TIMES; i++) {
    if (i == WARMUP_TIMES)
      checkCUDA(cudaEventRecord(startEvent));
    if (numDim == 2) {
      // Normal 2D Matmul
      checkCUDA(cublasSgemm(blas, transA, transB,
          m, n, k, &alpha, inputPtr, lda,
          filterPtr, ldb, &beta, outputPtr, ldc));
    } else {
      // Batched Matmul
      int strideA = mm->inputs[0].stride[numDim-3];
      int strideB = mm->inputs[1].stride[numDim-3];
      int strideC = mm->outputs[0].stride[numDim-3];
      int batch = 1;
      for (int i = 0; i < numDim-2; i++)
        batch *= mm->outputs[0].dim[i];
      checkCUDA(cublasSgemmStridedBatched(blas, transA, transB,
          m, n, k, &alpha, inputPtr, lda, strideA,
          filterPtr, ldb, strideB,
          &beta, outputPtr, ldc, strideC, batch));
    }
    if (mm->activation != AC_MODE_NONE)
      checkCUDNN(cudnnActivationForward(dnn, actiDesc,
          &alpha, outputTensor, outputPtr,
          &beta, outputTensor, outputPtr));
  } 
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  mm->runtime = milliseconds / REPEAT_TIMES;
  if (print_cost)
    printf("  measure[Matmul]: %s %s acti(%d) cost(%.4lf)\n",
           mm->inputs[0].to_string("input").c_str(),
           mm->inputs[1].to_string("weight").c_str(),
           mm->activation, mm->runtime);
}

