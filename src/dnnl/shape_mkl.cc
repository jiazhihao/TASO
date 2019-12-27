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

void Shape::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  CHECK_NE(nullptr, outputs[0].data_ptr = malloc(outputSize));

  if (type == OP_SHAPE) {
    assert(outputs[0].volume() == inputs[0].numDim);
    for (int i = 0; i < inputs[0].numDim; i++) {
      ((DATATYPE*)outputs[0].data_ptr)[i] = inputs[0].dim[i];
    }
  } else {
    assert(type == OP_SIZE);
    assert(outputs[0].volume() == 1);
    *((DATATYPE*)outputs[0].data_ptr) = inputs[0].volume();
  }
}

void Shape::unmap(void)
{
  // free tensors
  free(outputs[0].data_ptr);
  outputs[0].data_ptr = nullptr;
}

void Shape::forward(bool block)
{
  if (block) model->strm.wait();
}

void Model::measure_shape_cost(Shape* shape)
{
  // assume the cost is zero for now
  shape->runtime = 0;
}

