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

void Unsqueeze::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE);
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Unsqueeze::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Unsqueeze::forward(bool block)
{
  copy_kernel<<<GET_BLOCKS(outputs[0].volume()), CUDA_NUM_THREADS>>>(
      (float*)outputs[0].data_ptr, (float*)inputs[0].data_ptr, outputs[0].volume());
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_unsqueeze_cost(Unsqueeze* unsqz)
{
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    copy_kernel<<<GET_BLOCKS(unsqz->outputs[0].volume()), CUDA_NUM_THREADS>>>(
        outputPtr, inputPtr, unsqz->outputs[0].volume());
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  unsqz->runtime = milliseconds / REPEAT_TIMES;
  if (print_cost)
    printf("  measure[Squeeeze]: cost(%.4lf)\n", unsqz->runtime);
}
