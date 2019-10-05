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

TensorHandle Graph::reshape(const TensorHandle _input,
                            const std::vector<int>& _shape)
{
  std::vector<int> myshape = _shape;
  // replace zeros with input dims
  for (size_t i = 0; i < myshape.size(); i++)
    if (myshape[i] == 0)
      myshape[i] = _input->dim[i];
  int input_size = _input->volume();
  // replace -1 with actual size
  for (size_t i = 0; i < myshape.size(); i++)
    if (myshape[i] != -1) {
      assert(input_size % myshape[i] == 0);
      input_size /= myshape[i];
    }
  for (size_t i = 0; i < myshape.size(); i++)
    if (myshape[i] == -1) {
      myshape[i] = input_size;
      input_size = 1;
    }
  assert(input_size == 1);
  Op op = model->get_or_create_reshape(*_input, myshape);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_reshape(Tensor _input,
                                const std::vector<int>& _shape)
{
  ReshapeKey key(_input, _shape);
  Reshape* reshapeOp;
  if (reshape.find(key) != reshape.end()) {
    reshapeOp = reshape[key];
  } else {
    reshapeOp = new Reshape(this, _input, _shape);
    measure_reshape_cost(reshapeOp);
    reshape[key] = reshapeOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = reshapeOp;
  return ret;
}

Reshape::Reshape(Model* _model, Tensor _input,
                 const std::vector<int>& _shape)

: OpBase(_input, _model, OP_RESHAPE)
{
  int size = 1;
  // set dims and strides
  numOutputs = 1;
  outputs[0].numDim = _shape.size();
  for (int i = _shape.size() - 1; i >= 0; i--) {
    outputs[0].dim[i] = _shape[i];
    outputs[0].stride[i] = size;
    size *= _shape[i];
    outputs[0].split[i] = SplitInfo::NO_SPLIT;
  }
  assert(_input.volume() == size);
  outputs[0].idx = 0;
}

Reshape::~Reshape(void)
{}

bool Reshape::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void Reshape::collect_costs(float& exe_time, float& flops,
                            float& mem_acc, int& num_kernels)
{
}

ReshapeKey::ReshapeKey(Tensor _input, const std::vector<int>& shape)
{
  int idx = 0;
  keys[idx++] = shape.size();
  for (size_t i = 0; i < shape.size(); i++)
    keys[idx++] = shape[i];
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}

