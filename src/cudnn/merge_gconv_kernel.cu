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
void merge_gconv_kernel(DATATYPE* dst_ptr,
                        const DATATYPE* src_ptr,
                        int volume,
                        int c_in_h_w,
                        int c_out,
                        int count)
{
  assert(c_out % count == 0);
  CUDA_KERNEL_LOOP(i, volume)
  {
    int mod = i % c_in_h_w;
    int div = i / c_in_h_w;
    int dst_i = div * c_in_h_w * count + div / (c_out / count) * c_in_h_w + mod;
    dst_ptr[dst_i] = src_ptr[i];
  }
}

void MergeGConv::map(void)
{
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void MergeGConv::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void MergeGConv::forward(bool block)
{
  int c_out = inputs[0].dim[0];
  int c_in_h_w = inputs[0].volume() / c_out;
  assert(outputs[0].dim[1] % inputs[0].dim[1] == 0);
  int count = outputs[0].dim[1] / inputs[0].dim[1];
  assign_kernel<<<GET_BLOCKS(outputs[0].volume()), CUDA_NUM_THREADS>>>(
      (DATATYPE*)outputs[0].data_ptr, outputs[0].volume(), 0.0f);
  merge_gconv_kernel<<<GET_BLOCKS(inputs[0].volume()), CUDA_NUM_THREADS>>>(
      (DATATYPE*)outputs[0].data_ptr, (DATATYPE*)inputs[0].data_ptr,
      inputs[0].volume(), c_in_h_w, c_out, count);

  if (block)
    checkCUDA(cudaDeviceSynchronize());
}
