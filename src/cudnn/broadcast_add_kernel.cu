/* Copyright 2020 Stanford
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
void broadcast_add_kernel(int batch,
                          int channel,
                          int h_w_size,
                          DATATYPE* dst_ptr,
                          DATATYPE* _data,
                          DATATYPE* _bias)
{
  int volume = batch * channel * h_w_size;
  CUDA_KERNEL_LOOP(i, volume)
  {
    int channel_idx = i % h_w_size;
    dst_ptr[i] = _data[i] + _bias[channel_idx];
  }
}

void BroadcastAdd::map(void)
{
  assert(inputs[0].numDim == 4);
  assert(inputs[1].numDim == 1);
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void BroadcastAdd::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void BroadcastAdd::forward(bool block)
{
  int batch = outputs[0].dim[0];
  int channel = outputs[0].dim[1];
  int h_w_size = outputs[0].dim[2] * outputs[0].dim[3];
  DATATYPE* _data_ptr = (DATATYPE*) inputs[0].data_ptr;
  DATATYPE* _bias_ptr = (DATATYPE*) inputs[1].data_ptr;
  broadcast_add_kernel<<<GET_BLOCKS(outputs[0].volume()), CUDA_NUM_THREADS>>>(
      batch, channel, h_w_size, (DATATYPE*)outputs[0].data_ptr,
      _data_ptr, _bias_ptr);
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}


