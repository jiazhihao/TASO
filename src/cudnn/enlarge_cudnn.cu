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

#include "xflow/ops.h"
#include "xflow/cuda_helper.h"
using namespace XFlow;

__global__
void enlarge_kernel(DATATYPE* dst_ptr,
                    const DATATYPE* src_ptr,
                    int volume,
                    int dst_h,
                    int dst_w,
                    int src_h,
                    int src_w)
{
  int off_h = (dst_h - src_h) / 2;
  int off_w = (dst_w - src_w) / 2;
  CUDA_KERNEL_LOOP(i, volume)
  {
    int h = (i % (dst_h * dst_w)) / dst_w - off_h;
    int w = (i % (dst_h * dst_w)) % dst_w - off_w;
    if ((h < 0) || (h >= src_h) || (w < 0) || (w >= src_w))
      dst_ptr[i] = 0.0f;
    else {
      int offset = (i / (dst_h * dst_w)) * (src_h * src_w) + h * src_w + w;
      dst_ptr[i] = src_ptr[offset];
    }
  }
}

void Enlarge::map(void)
{
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Enlarge::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Enlarge::forward(bool block)
{
  enlarge_kernel<<<GET_BLOCKS(outputs[0].volume()), CUDA_NUM_THREADS>>>(
      (DATATYPE*)outputs[0].data_ptr, (DATATYPE*)inputs[0].data_ptr, outputs[0].volume(),
      outputs[0].dim[2], outputs[0].dim[3], inputs[0].dim[2], inputs[0].dim[3]);
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_enlarge_cost(Enlarge* enl)
{
  enl->runtime = 0.0f;
}
