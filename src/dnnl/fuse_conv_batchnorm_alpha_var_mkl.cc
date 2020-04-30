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

void fuse_conv_batchnorm_alpha_var_kernel(int c_out, int c_in_h_w,
    DATATYPE* dst_ptr, const DATATYPE* conv_w, const DATATYPE* scale, const DATATYPE* var) {
  int volume = c_out * c_in_h_w;
#pragma omp parallel for
  for (int i = 0; i < volume; i++) {
    int c_out_idx = i / c_in_h_w;
    dst_ptr[i] = scale[c_out_idx] * conv_w[i] / std::sqrt(std::abs(var[c_out_idx]) + BN_MIN_EPSILON);
  }
}

void FuseConvBatchNormAlphaVar::map(void)
{
  assert(inputs[0].numDim == 4);
  assert(inputs[1].numDim == 1);
  assert(inputs[2].numDim == 1);
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  CHECK_NE(nullptr, outputs[0].data_ptr = malloc(outputSize));
}

void FuseConvBatchNormAlphaVar::unmap(void)
{
  // clear primitives
  net.clear();
  // free tensors
  free(outputs[0].data_ptr);
  outputs[0].data_ptr = nullptr;
}

void FuseConvBatchNormAlphaVar::forward(bool block)
{
  int c_out = outputs[0].dim[0];
  int c_in_h_w = outputs[0].volume() / c_out;
  DATATYPE* conv_w_ptr = (DATATYPE*) inputs[0].data_ptr;
  DATATYPE* scale_ptr = (DATATYPE*) inputs[1].data_ptr;
  DATATYPE* var_ptr = (DATATYPE*) inputs[2].data_ptr;
  fuse_conv_batchnorm_alpha_var_kernel(c_out, c_in_h_w,
      (DATATYPE*)outputs[0].data_ptr, conv_w_ptr, scale_ptr, var_ptr);
}

