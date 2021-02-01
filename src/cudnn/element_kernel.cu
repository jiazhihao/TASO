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

__global__
void elementwise_kernel(int volume,
                        OpType type,
                        const Tensor xTensor,
                        const Tensor yTensor,
                        const Tensor zTensor,
                        const DATATYPE* x,
			const DATATYPE* y,
			DATATYPE* z)
{
  int pos[6];
  assert(zTensor.numDim <= 6);
  CUDA_KERNEL_LOOP(id_z, volume)
  {
    int id_x = 0, id_y = 0;
    for (int j = 0; j < zTensor.numDim; j++) {
      pos[j] = (id_z / zTensor.stride[j]) % zTensor.dim[j];
    }
    int diff = zTensor.numDim - xTensor.numDim;
    for (int j = 0; j < xTensor.numDim; j++) {
      id_x += xTensor.stride[j] * pos[j + diff];
    }
    diff = zTensor.numDim - yTensor.numDim;
    for (int j = 0; j < yTensor.numDim; j++) {
      id_y += yTensor.stride[j] * pos[j + diff];
    }

    switch (type) {
      case OP_EW_ADD:
      {
        z[id_z] = x[id_x] + y[id_y];
        break;
      }
      case OP_EW_MUL:
      {
        z[id_z] = x[id_x] * y[id_y];
        break;
      }
      case OP_EW_MAX:
      {
        z[id_z] = max(x[id_x], y[id_y]);
        break;
      }
      case OP_EW_MIN:
      {
        z[id_z] = min(x[id_x], y[id_y]);
        break;
      }
      case OP_EW_SUB:
      {
        z[id_z] = x[id_x] - y[id_y];
        break;
      }
      case OP_EW_DIV:
      {
        z[id_z] = x[id_x] / y[id_y];
        break;
      }
      case OP_EW_EQUAL:
      {
        z[id_z] = (x[id_x] == y[id_y]);
        break;
      }
      case OP_EW_GREATER:
      {
        z[id_z] = (x[id_x] > y[id_y]);
        break;
      }
      case OP_EW_LESS:
      {
        z[id_z] = (x[id_x] < y[id_y]);
        break;
      }
      case OP_PRELU:
      {
        z[id_z] = x[id_x] >= 0 ? x[id_x] : y[id_y] * x[id_x];
        break;
      }
      default:
        assert(false);
    }
  }
}

bool Element::use_kernel(void) const
{
  switch (type) {
    case OP_EW_ADD:
    case OP_EW_MUL:
    case OP_EW_MAX:
    case OP_EW_MIN:
      break;
    default:
      return false;
  }
  // use cudnn kernel only if inputs and output have default layouts
  if (inputs[0].default_layout() && inputs[1].default_layout()
  && outputs[0].default_layout()) {
    // Do nothing
  } else {
    return false;
  }
  // CUDNNv7.6.5 raequirement: Each dimension of the input tensor A must 
  // match the corresponding dimension of the destination tensor C, and 
  // each dimension of the input tensor B must match the corresponding 
  // dimension of the destination tensor C or must be equal to 1. 
  // In the latter case, the same value from the input tensor B for 
  // those dimensions will be used to blend into the C tensor.
  if (inputs[0].volume() != outputs[0].volume())
    return false;
  return true;
}

void Element::map(void)
{
  if (use_kernel()) {
    // create descriptors
    checkCUDNN(cudnnCreateTensorDescriptor(&in1Tensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&in2Tensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&outTensor));
    checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
    // set descriptors
    helperSetBroadcastableTensorDescriptor(inputs[0], outputs[0], in1Tensor);
    helperSetBroadcastableTensorDescriptor(inputs[1], outputs[0], in2Tensor);
    helperSetTensorDescriptor(outputs[0], outTensor);

    cudnnOpTensorOp_t opType;
    switch (type) {
      case OP_EW_ADD:
        opType = CUDNN_OP_TENSOR_ADD;
        break;
      case OP_EW_MUL:
        opType = CUDNN_OP_TENSOR_MUL;
        break;
      case OP_EW_MAX:
        opType = CUDNN_OP_TENSOR_MAX;
        break;
      case OP_EW_MIN:
        opType = CUDNN_OP_TENSOR_MIN;
        break;
      default:
        fprintf(stderr, "Unsupported Elementwise Operator by cuDNN: %d\n", type);
        assert(false);
    }
    checkCUDNN(cudnnSetOpTensorDescriptor(opDesc, opType, CUDNN_DATA_FLOAT,
        CUDNN_NOT_PROPAGATE_NAN));
  } else {
    // No preprocessing for our customized kernel
  }
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE);
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Element::unmap(void)
{
  if (use_kernel()) {
    checkCUDNN(cudnnDestroyTensorDescriptor(in1Tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(in2Tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(outTensor));
    checkCUDNN(cudnnDestroyOpTensorDescriptor(opDesc));
  }
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Element::forward(bool block)
{
  if (use_kernel()) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    checkCUDNN(cudnnOpTensor(model->dnn, opDesc, &alpha, in1Tensor, inputs[0].data_ptr,
        &alpha, in2Tensor, inputs[1].data_ptr, &beta, outTensor, outputs[0].data_ptr));
  } else {
    elementwise_kernel<<<GET_BLOCKS(outputs[0].volume()), CUDA_NUM_THREADS>>>(
        outputs[0].volume(), type, inputs[0], inputs[1], outputs[0],
        (DATATYPE*)inputs[0].data_ptr, (DATATYPE*)inputs[1].data_ptr,
	(DATATYPE*)outputs[0].data_ptr);
  }
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_element_cost(Element* ele)
{
  // cudnnOpTensor only supports OP_EW_ADD, OP_EW_MUL, OP_EW_MAX, OP_EW_MIN
  if (ele->use_kernel()) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    helperSetBroadcastableTensorDescriptor(ele->inputs[0],
        ele->outputs[0], inputTensor);
    helperSetBroadcastableTensorDescriptor(ele->inputs[1],
        ele->outputs[0], biasTensor);
    helperSetTensorDescriptor(ele->outputs[0], outputTensor);
    cudnnOpTensorOp_t opType;
    switch (ele->type) {
      case OP_EW_ADD:
        opType = CUDNN_OP_TENSOR_ADD;
        break;
      case OP_EW_MUL:
        opType = CUDNN_OP_TENSOR_MUL;
        break;
      case OP_EW_MAX:
        opType = CUDNN_OP_TENSOR_MAX;
        break;
      case OP_EW_MIN:
        opType = CUDNN_OP_TENSOR_MIN;
        break;
      default:
      {
        fprintf(stderr, "Unsupported Elementwise Operator by cuDNN: %d\n", ele->type);
        assert(false);
      }
    }
    checkCUDNN(cudnnSetOpTensorDescriptor(opDesc, opType, CUDNN_DATA_FLOAT,
        CUDNN_NOT_PROPAGATE_NAN));
  
    checkCUDA(cudaDeviceSynchronize());
    checkCUDA(cudaEventRecord(startEvent));
    for (int i = 0; i < REPEAT_TIMES; i++) {
      checkCUDNN(cudnnOpTensor(dnn, opDesc, &alpha, inputTensor, inputPtr,
          &alpha, biasTensor, filterPtr, &beta, outputTensor, outputPtr));
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
  } else {
    // Use our implementation to measure other elementwise operators
    checkCUDA(cudaDeviceSynchronize());
    checkCUDA(cudaEventRecord(startEvent));
    for (int i = 0; i < REPEAT_TIMES; i++) {
      elementwise_kernel<<<GET_BLOCKS(ele->outputs[0].volume()), CUDA_NUM_THREADS>>>(
          ele->outputs[0].volume(), ele->type, ele->inputs[0], ele->inputs[1],
          ele->outputs[0], inputPtr, filterPtr, outputPtr);
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
}

