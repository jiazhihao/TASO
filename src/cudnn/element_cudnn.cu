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

#include "xflow/ops.h"
#include "xflow/cuda_helper.h"
using namespace XFlow;

void Element::map(void)
{
  // create descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
  // set descriptors
  helperSetTensorDescriptor(inputs[0], inputTensor);

  cudnnOpTensorOp_t opType;
  switch (type) {
    case OP_EW_ADD:
      opType = CUDNN_OP_TENSOR_ADD;
      break;
    case OP_EW_MUL:
      opType = CUDNN_OP_TENSOR_MUL;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetOpTensorDescriptor(opDesc, opType, CUDNN_DATA_FLOAT,
      CUDNN_NOT_PROPAGATE_NAN));
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE);
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Element::unmap(void)
{
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyOpTensorDescriptor(opDesc));
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Element::forward(bool block)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  checkCUDNN(cudnnOpTensor(model->dnn, opDesc, &alpha, inputTensor, inputs[0].data_ptr,
      &alpha, inputTensor, inputs[1].data_ptr, &beta, inputTensor, outputs[0].data_ptr));
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_element_cost(Element* ele)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  helperSetTensorDescriptor(ele->inputs[0], inputTensor);
  //int inputN = ele->inputs[0].dim[0];
  //int inputC = max(ele->inputs[0].dim[1], 1);
  //int inputH = max(ele->inputs[0].dim[2], 1);
  //int inputW = max(ele->inputs[0].dim[3], 1);
  //checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW,
  //    CUDNN_DATA_FLOAT, inputN, inputC, inputH, inputW));

  cudnnOpTensorOp_t opType;
  switch (ele->type) {
    case OP_EW_ADD:
      opType = CUDNN_OP_TENSOR_ADD;
      break;
    case OP_EW_MUL:
      opType = CUDNN_OP_TENSOR_MUL;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetOpTensorDescriptor(opDesc, opType, CUDNN_DATA_FLOAT,
      CUDNN_NOT_PROPAGATE_NAN));

  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    checkCUDNN(cudnnOpTensor(dnn, opDesc, &alpha, inputTensor, inputPtr,
        &alpha, inputTensor, filterPtr, &beta, inputTensor, outputPtr));
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  ele->runtime = milliseconds / REPEAT_TIMES;
  if (print_cost)
    printf("  measure[Element]: i(%d %d %d %d) type(%d) cost(%.4lf)\n",
           ele->inputs[0].dim[0], ele->inputs[0].dim[1], ele->inputs[0].dim[2],
           ele->inputs[0].dim[3], ele->type, ele->runtime);
}

