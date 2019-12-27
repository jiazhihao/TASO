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

void enlarge_kernel(DATATYPE* dstPtr, const DATATYPE* srcPtr, int volume,
    const int dstH, const int dstW, const int srcH, const int srcW) {
  int offH = (dstH - srcH) / 2;
  int offW = (dstW - srcW) / 2;
#pragma omp parallel for
  for (int i = 0; i < volume; i++) {
    int h = (i % (dstH * dstW)) / dstW - offH;
    int w = (i % (dstH * dstW)) % dstW - offW;
    if (h < 0 || h >= srcH || w < 0 || w >= srcW)
      dstPtr[i] = 0.0f;
    else {
      int offset = (i / (dstH * dstW)) * (srcH * srcW) + h * srcW + w;
      dstPtr[i] = srcPtr[offset];
    }
  }
}

void Enlarge::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  CHECK_NE(nullptr, outputs[0].data_ptr = malloc(outputSize));
}

void Enlarge::unmap(void)
{
  // clear primitives
  net.clear();
  // free tensors
  free(outputs[0].data_ptr);
  outputs[0].data_ptr = nullptr;
}

void Enlarge::forward(bool block)
{
  enlarge_kernel((DATATYPE*)outputs[0].data_ptr, (DATATYPE*)inputs[0].data_ptr, outputs[0].volume(),
      outputs[0].dim[2], outputs[0].dim[3], inputs[0].dim[2], inputs[0].dim[3]);
}

void Model::measure_enlarge_cost(Enlarge* enl)
{
  // measure.
  uint64_t beg = 0;
  for (int i = 0; i < WARMUP_TIMES + REPEAT_TIMES; i++) {
    if (i == WARMUP_TIMES) {
      beg = microsecond_timer();
    }
    enlarge_kernel(outputPtr, inputPtr, enl->outputs[0].volume(),
      enl->outputs[0].dim[2], enl->outputs[0].dim[3],
      enl->inputs[0].dim[2], enl->inputs[0].dim[3]);
  }
  auto end = microsecond_timer();

  enl->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // milliseconds
  if (print_cost)
    printf("  measure[Enlarge]: cost(%.4lf)\n", enl->runtime);
}

