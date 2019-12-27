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

void Reshape::map(void)
{
  // allocate tensors
  assert(outputs[0].volume() == inputs[0].volume());
  // for now the output and input share the same instance
  outputs[0].data_ptr = inputs[0].data_ptr;
}

void Reshape::unmap(void)
{
  // clear primitives
  net.clear();
}

void Reshape::forward(bool block)
{
  if (block) model->strm.wait();
}

void Model::measure_reshape_cost(Reshape* reshape)
{
  // Reshape requires no kernel launch
  reshape->runtime = 0;
  if (print_cost)
    printf("  measure[Reshape]: cost(%.4lf)\n", reshape->runtime);
}

