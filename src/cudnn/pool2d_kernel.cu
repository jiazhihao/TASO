/* Copyright 2018 Stanford
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

void Pool2D::map(void)
{
  // create descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
  int inputC = inputs[0].dim[1];
  int inputH = inputs[0].dim[2];
  int inputW = inputs[0].dim[3];
  int padH, padW;
  get_padding(&padH, &padW);
  // set descriptors
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, BATCH_SIZE, inputC, inputH, inputW));
  cudnnPoolingMode_t mode;
  if (type == OP_POOL2D_MAX)
    mode = CUDNN_POOLING_MAX;
  else if (type == OP_POOL2D_AVG)
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, mode, CUDNN_PROPAGATE_NAN,
      kernelH, kernelW, padH, padW, strideH, strideW));
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, 
      inputTensor, &n, &c, &h, &w));
  assert(n == BATCH_SIZE);
  assert(c == inputC);
  assert(outputs[0].dim[2] == h);
  assert(outputs[0].dim[3] == w);
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, n, c, h, w));
  if (activation != AC_MODE_NONE) {
    checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    cudnnActivationMode_t mode = get_activation_mode(activation);
    checkCUDNN(cudnnSetActivationDescriptor(actiDesc, mode,
        CUDNN_PROPAGATE_NAN, 0.0));
  }
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * n * c * h * w;
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Pool2D::unmap(void)
{
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
  if (activation != AC_MODE_NONE) {
    checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  }
  // free tensors
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Pool2D::forward(bool block)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  checkCUDNN(cudnnPoolingForward(model->dnn, poolDesc,
      &alpha, inputTensor, inputs[0].data_ptr,
      &beta, outputTensor, outputs[0].data_ptr));
  if (activation != AC_MODE_NONE) {
    checkCUDNN(cudnnActivationForward(model->dnn, actiDesc,
        &alpha, outputTensor, outputs[0].data_ptr,
        &beta, outputTensor, outputs[0].data_ptr));
  }
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_pool2d_cost(Pool2D* pool)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int inputC = pool->inputs[0].dim[1];
  int inputH = pool->inputs[0].dim[2];
  int inputW = pool->inputs[0].dim[3];
  int outputH = pool->outputs[0].dim[2];
  int outputW = pool->outputs[0].dim[3];
  int padH, padW;
  // Reference: https://www.tensorflow.org/api_guides/python/nn#Convolution
  switch (pool->padding) {
    case PD_MODE_SAME:
      int totalPadH, totalPadW;
      if (inputH % pool->strideH == 0)
        totalPadH = max(pool->kernelH - pool->strideH, 0);
      else
        totalPadH = max(pool->kernelH - (inputH % pool->strideH), 0);
      if (inputW % pool->strideW == 0)
        totalPadW = max(pool->kernelW - pool->strideW, 0);
      else
        totalPadW = max(pool->kernelW - (inputW % pool->strideW), 0);
      // assert same padding on both sides
      padH = (totalPadH + 1) / 2;
      padW = (totalPadW + 1)/ 2;
      break;
    case PD_MODE_VALID:
      padH = 0;
      padW = 0;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, BATCH_SIZE, inputC, inputH, inputW));
  cudnnPoolingMode_t mode;
  if (pool->type == OP_POOL2D_MAX)
    mode = CUDNN_POOLING_MAX;
  else if (pool->type == OP_POOL2D_AVG)
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, mode,
      CUDNN_PROPAGATE_NAN, pool->kernelH, pool->kernelW, padH, padW,
      pool->strideH, pool->strideW));
  checkCUDNN(cudnnSetActivationDescriptor(actiDesc, CUDNN_ACTIVATION_RELU,
        CUDNN_NOT_PROPAGATE_NAN, 0.0));
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc,
      inputTensor, &n, &c, &h, &w));
  assert(n == BATCH_SIZE);
  assert(c == inputC);
  assert(outputH == h);
  assert(outputW == w);
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, n, c, h, w));
  size_t inputSize = sizeof(DATATYPE) * BATCH_SIZE * inputC * inputH * inputW;
  size_t outputSize = sizeof(DATATYPE) * BATCH_SIZE * inputC * outputH * outputW;
  assert(inputSize < MAX_TENSOR_SIZE);
  assert(outputSize < MAX_TENSOR_SIZE);
  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < WARMUP_TIMES + REPEAT_TIMES; i++) {
    if (i == WARMUP_TIMES) {
      checkCUDA(cudaEventRecord(startEvent));
    }
    checkCUDNN(cudnnPoolingForward(dnn, poolDesc,
        &alpha, inputTensor, inputPtr,
        &beta, outputTensor, outputPtr));
    if (pool->activation != AC_MODE_NONE) {
      checkCUDNN(cudnnActivationForward(dnn, actiDesc,
          &alpha, outputTensor, outputPtr,
          &beta, outputTensor, outputPtr));
    }
    // Backward computation
    checkCUDNN(cudnnPoolingBackward(dnn, poolDesc,
        &alpha, outputTensor, outputPtr,
        outputTensor, outputPtr,
        inputTensor, inputPtr,
        &beta, inputTensor, inputPtr));
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  pool->runtime = milliseconds / REPEAT_TIMES;
  if (print_cost)
    printf("  measure[Pool2D]: i(%d %d %d %d) k(%d %d) s(%d %d) p(%d %d) cost(%.4lf)\n",
           BATCH_SIZE, inputC, inputH, inputW, pool->kernelH, pool->kernelW,
           pool->strideH, pool->strideW, padH, padW, pool->runtime);
}

