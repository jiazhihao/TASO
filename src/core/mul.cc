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

#include "xflow/ops.h"
using namespace XFlow;

TensorHandle Graph::mul(const TensorHandle x,
                        const TensorHandle y)
{
  Op op = model->get_or_create_mul(*x, *y);
  add_edge(x->op, op, x->idx, 0);
  add_edge(y->op, op, y->idx, 1);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_mul(const Tensor& x,
                            const Tensor& y)
{
  MulKey key(x, y);
  Mul* mulOp;
  if (mul.find(key) != mul.end()) {
    mulOp = mul[key];
  } else {
    mulOp = new Mul(this, x, y);
    measure_mul_cost(mulOp);
    mul[key] = mulOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = mulOp;
  return ret;
}

Mul::Mul(Model* _model, const Tensor& x, const Tensor& y)
: OpBase(x, y, _model, OP_MUL)
{
  // TODO: support broadcast
  // Currently assume _y.numDim = 0
  int numDim = x.numDim;
  assert(y.numDim == 0);
  for (int i = 0; i < y.numDim; i++)
    assert(x.dim[i] == y.dim[i]);
  numOutputs = 1;
  outputs[0].numDim = numDim;
  for (int i = 0; i < numDim-1; i++) {
    outputs[0].dim[i] = x.dim[i];
    outputs[0].stride[i] = x.stride[i];
    outputs[0].split[i] = x.split[i];
  }
  outputs[0].idx = 0;
}

Mul::~Mul(void)
{}

bool Mul::get_parameter(PMParameter para, int* value)
{
  return OpBase::get_parameter(para, value);
}

void Mul::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  // TODO: to be implemented
  assert(false);
}

MulKey::MulKey(const Tensor& _x, const Tensor& _y)
{
  int idx = 0;
  _x.serialize(keys, idx);
  _y.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}

