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
using namespace taso;

// Enlarge the third and forth dimension of _w1 to the same size as _w2
TensorHandle Graph::enlarge(const TensorHandle _w1,
                            const TensorHandle _w2)
{
  // Currently the weight being enlarged must be 4D:
  // Cout, Cin, KerelH, KernelW
  assert(_w1->numDim == 4);
  assert(_w2->numDim == 4);
  assert(_w1->dim[2] <= _w2->dim[2]);
  assert(_w1->dim[3] <= _w2->dim[3]);
  Op op = model->get_or_create_enlarge(*_w1, *_w2);
  assert(op != Op::INVALID_OP);
  add_edge(_w1->op, op, _w1->idx, 0);
  add_edge(_w2->op, op, _w2->idx, 1);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_enlarge(Tensor _w1, Tensor _w2)
{
  // Check 1: w1 and w2 must both have 4D
  if (_w1.numDim != 4 || _w2.numDim != 4)
    return Op::INVALID_OP;
  // Check 2: w1 is smaller than w2
  if (_w1.dim[2] > _w2.dim[2] || _w1.dim[3] > _w2.dim[3])
    return Op::INVALID_OP;
  EnlargeKey key(_w1, _w2);
  Enlarge* enlargeOp;
  if (enlarge.find(key) != enlarge.end()) {
    enlargeOp = enlarge[key];
  } else {
    enlargeOp = new Enlarge(this, _w1, _w2);
    measure_enlarge_cost(enlargeOp);
    enlarge[key] = enlargeOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = enlargeOp;
  return ret;
}

Enlarge::Enlarge(Model* _model, Tensor _w1, Tensor _w2)
: OpBase(_w1, _w2, _model, OP_ENLARGE)
{
  assert(_w1.numDim == 4);
  assert(_w2.numDim == 4);
  assert(_w1.dim[2] <= _w2.dim[2]);
  assert(_w1.dim[3] <= _w2.dim[3]);
  numOutputs = 1;
  outputs[0].numDim = _w1.numDim;
  outputs[0].dim[0] = _w1.dim[0];
  outputs[0].dim[1] = _w1.dim[1];
  outputs[0].dim[2] = _w2.dim[2];
  outputs[0].dim[3] = _w2.dim[3];
  outputs[0].stride[3] = 1;
  outputs[0].stride[2] = outputs[0].stride[3] * outputs[0].dim[3];
  outputs[0].stride[1] = outputs[0].stride[2] * outputs[0].dim[2];
  outputs[0].stride[0] = outputs[0].stride[1] * outputs[0].dim[1];
  // Set SplitInfo
  outputs[0].split[0] = _w1.split[0];
  outputs[0].split[1] = _w1.split[1];
  outputs[0].idx = 0;
}

Enlarge::~Enlarge(void)
{}

bool Enlarge::get_parameter(PMParameter para, int* value)
{
  switch (para) {
    //case PM_KERNEL_H:
    //  *value = kernelH;
    //  return true;
    //case PM_KERNEL_W:
    //  *value = kernelW;
    //  return true;
    default:
      return OpBase::get_parameter(para, value);
  }
}

void Enlarge::collect_costs(float& exe_time, float& flops,
                            float& mem_acc, int& num_kernels)
{
  int outputSize = outputs[0].volume();
  int inputSize = inputs[0].volume();
  exe_time += runtime;
  flops += outputSize;
  mem_acc += inputSize + outputSize;
  num_kernels += 1;
}

// keys are (kernelH, kernelW, _weight)
EnlargeKey::EnlargeKey(Tensor _w1, Tensor _w2)
{
  assert(_w1.numDim == 4);
  int idx = 0;
  _w1.serialize(keys, idx);
  _w2.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}
