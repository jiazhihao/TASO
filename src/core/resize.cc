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

TensorHandle Graph::resize(const TensorHandle _input,
                           const std::vector<int>& _shape)
{
  Op op = model->get_or_create_resize(*_input, _shape);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_resize(const Tensor& _input,
                               const std::vector<int>& _shape)
{
  if ((int)_shape.size() != _input.numDim)
    return Op::INVALID_OP;
  ResizeKey key(_input, _shape);
  Resize* resizeOp;
  if (resize.find(key) != resize.end()) {
    resizeOp = resize[key];
  } else {
    resizeOp = new Resize(this, _input, _shape);
    measure_resize_cost(resizeOp);
    resize[key] = resizeOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = resizeOp;
  return ret;
}

Resize::Resize(Model* _model, const Tensor& _input,
               const std::vector<int>& _shape)
: OpBase(_input, _model, OP_RESIZE), shape(_shape)
{
  assert((int)_shape.size() == _input.numDim);
  numOutputs = 1;
  outputs[0].numDim = _input.numDim;
  // Currently assume the input tensor has the default layout
  assert(_input.default_layout());
  int total = 1;
  for (int i = _input.numDim-1; i >= 0; i--) {
    outputs[0].dim[i] = _shape[i];
    if (_shape[i] == _input.dim[i])
      outputs[0].split[i] = _input.split[i];
    else
      outputs[0].split[i] = SplitInfo::NO_SPLIT;
    outputs[0].stride[i] = total;
    total *= outputs[0].dim[i];
  }
  outputs[0].idx = 0;
}

Resize::~Resize(void)
{}

bool Resize::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void Resize::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  exe_time += runtime;
}

ResizeKey::ResizeKey(const Tensor& _input,
                     const std::vector<int>& _shape)
{
  int idx = 0;
  keys[idx++] = _shape.size();
  for (size_t i = 0; i < _shape.size(); i++)
    keys[idx++] = _shape[i];
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}
