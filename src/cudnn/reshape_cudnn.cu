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

void Reshape::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Reshape::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Reshape::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_reshape_cost(Reshape* reshape)
{
  // FIXME: assume the cost is zero for now
  reshape->runtime = 0;
}
