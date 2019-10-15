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

void Activation::map(void)
{
  // create descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  helperSetTensorDescriptor(inputs[0], inputTensor);
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  cudnnActivationMode_t mode;
  switch (type) {
    case OP_RELU:
      mode = CUDNN_ACTIVATION_RELU;
      break;
    case OP_SIGMOID:
      mode = CUDNN_ACTIVATION_SIGMOID;
      break;
    case OP_TANH:
      mode = CUDNN_ACTIVATION_TANH;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetActivationDescriptor(actiDesc, mode,
      CUDNN_NOT_PROPAGATE_NAN, 0.0));
  if (!inPlace) {
    size_t outputSize = sizeof(DATATYPE);
    for (int i = 0; i < inputs[0].numDim; i++)
      outputSize *= inputs[0].dim[i];
    checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
  } else {
    outputs[0].data_ptr = inputs[0].data_ptr;
  }
}

void Activation::unmap(void)
{
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  if (!inPlace) {
    checkCUDA(cudaFree(outputs[0].data_ptr));
  }
}

void Activation::forward(bool block)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  checkCUDNN(cudnnActivationForward(model->dnn, actiDesc,
      &alpha, inputTensor, inputs[0].data_ptr,
      &beta, inputTensor, outputs[0].data_ptr));
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_activation_cost(Activation* act)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  helperSetTensorDescriptor(act->inputs[0], inputTensor);
  cudnnActivationMode_t mode;
  switch (act->type) {
    case OP_RELU:
      mode = CUDNN_ACTIVATION_RELU;
      break;
    case OP_SIGMOID:
      mode = CUDNN_ACTIVATION_SIGMOID;
      break;
    case OP_TANH:
      mode = CUDNN_ACTIVATION_TANH;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetActivationDescriptor(actiDesc, mode,
      CUDNN_NOT_PROPAGATE_NAN, 0.0));
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    if (act->inPlace) {
      checkCUDNN(cudnnActivationForward(dnn, actiDesc,
          &alpha, inputTensor, inputPtr,
          &beta, inputTensor, inputPtr));
    } else {
      checkCUDNN(cudnnActivationForward(dnn, actiDesc,
          &alpha, inputTensor, inputPtr,
          &beta, inputTensor, outputPtr));
    }
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  act->runtime = milliseconds / REPEAT_TIMES;
  if (print_cost)
    printf("  measure[Activation]: i(%d %d %d %d) type(%d) cost(%.4lf)\n",
           act->inputs[0].dim[0], act->inputs[0].dim[1], act->inputs[0].dim[2],
           act->inputs[0].dim[3], act->type, act->runtime);
}

