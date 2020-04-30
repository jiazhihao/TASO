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

void fuse_conv_batchnorm_bias_kernel(int volume, DATATYPE* dst_ptr,
    const DATATYPE* scale, const DATATYPE* beta,
    const DATATYPE* mean, const DATATYPE* var) {
#pragma omp parallel for
  for (int i = 0; i < volume; i++) {
    dst_ptr[i] = beta[i] - scale[i] * mean[i] / std::sqrt(var[i] + BN_MIN_EPSILON);
  }
}

void FuseConvBatchNormBias::map(void)
{
  assert(inputs[0].numDim == 1);
  assert(inputs[1].numDim == 1);
  assert(inputs[2].numDim == 1);
  assert(inputs[3].numDim == 1);
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  CHECK_NE(nullptr, outputs[0].data_ptr = malloc(outputSize));
}

void FuseConvBatchNormBias::unmap(void)
{
  // clear primitives
  net.clear();
  // free tensors
  free(outputs[0].data_ptr);
  outputs[0].data_ptr = nullptr;
}

void FuseConvBatchNormBias::forward(bool block)
{
  int volume = outputs[0].volume();
  DATATYPE* scale_ptr = (DATATYPE*) inputs[0].data_ptr;
  DATATYPE* beta_ptr = (DATATYPE*) inputs[1].data_ptr;
  DATATYPE* mean_ptr = (DATATYPE*) inputs[2].data_ptr;
  DATATYPE* var_ptr = (DATATYPE*) inputs[3].data_ptr;
  fuse_conv_batchnorm_bias_kernel(volume,
      (DATATYPE*)outputs[0].data_ptr, scale_ptr, beta_ptr, mean_ptr, var_ptr);
}

