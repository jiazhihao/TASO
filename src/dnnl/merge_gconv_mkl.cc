/* Copyright 2020 Stanford, Tsinghua
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
#include "taso/dnnl_helper.h"
using namespace taso;
using namespace dnnl;

void merge_gconv_kernel(DATATYPE* dstPtr, const DATATYPE* srcPtr, int volume,
    int cInHW, int cOut, int count) {
  assert(cOut % count == 0);
#pragma omp parallel for
  for (int srcIdx = 0; srcIdx < volume; srcIdx++) {
    int mod = srcIdx % cInHW;
    int div = srcIdx / cInHW;
    int dstIdx = (div * count + div / (cOut / count)) * cInHW + mod;
    dstPtr[dstIdx] = srcPtr[srcIdx];
  }
}

void MergeGConv::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  CHECK_NE(nullptr, outputs[0].data_ptr = malloc(outputSize));
}

void MergeGConv::unmap(void)
{
  // clear primitives
  net.clear();
  // free tensors
  free(outputs[0].data_ptr);
  outputs[0].data_ptr = nullptr;
}

void MergeGConv::forward(bool block)
{
  // outChannels unchanged, inChannels enlarged by count.
  assert(outputs[0].dim[0] == inputs[0].dim[0]);
  int cOut = inputs[0].dim[0];
  int cInHW = inputs[0].volume() / cOut;
  assert(outputs[0].dim[1] % inputs[0].dim[1] == 0);
  int count = outputs[0].dim[1] / inputs[0].dim[1];

  assign_kernel((DATATYPE*)outputs[0].data_ptr, outputs[0].volume(), 0.0f);
  merge_gconv_kernel((DATATYPE*)outputs[0].data_ptr, (DATATYPE*)inputs[0].data_ptr,
      inputs[0].volume(), cInHW, cOut, count);

  if (block) model->strm.wait();
}

