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

void TopK::map(void)
{
  // TODO: use cudnn reduce tensor
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputs[0].volume() * sizeof(DATATYPE)));
  checkCUDA(cudaMalloc(&outputs[1].data_ptr, outputs[1].volume() * sizeof(DATATYPE)));
}

void TopK::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
  checkCUDA(cudaFree(outputs[1].data_ptr));
}

void TopK::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_topk_cost(TopK* topk)
{
  // TODO: use cudnn reduce tensor
  topk->runtime = 0;
}
