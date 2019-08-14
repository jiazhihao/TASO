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

TensorHandle Graph::constant(int ndim, int* dims, OpType type)
{
  Op op = model->get_or_create_constant(ndim, dims, type);
  // NOTE that constant do not have any inputs
  // we need to manually add op to the inedges
  assert(inEdges.find(op) == inEdges.end());
  inEdges[op];
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_constant(int ndim, int* dims, OpType _type)
{
  ConstantKey key(ndim, dims, _type);
  Constant* constantOp;
  if (constant.find(key) != constant.end()) {
    constantOp = constant[key];
  } else {
    constantOp = new Constant(this, ndim, dims, _type);
    constantOp->runtime = 0.0f;
    constant[key] = constantOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = constantOp;
  return ret;
}

Constant::Constant(Model* _model, int ndim, int* dims, OpType _type)
: OpBase(_model, _type)
{
  numOutputs = 1;
  outputs[0].numDim = ndim;
  for (int i = 0; i < ndim; i++)
    outputs[0].dim[i] = dims[i];
  outputs[0].stride[ndim-1] = 1;
  for (int i = ndim-2; i >= 0; i--)
    outputs[0].stride[i] = outputs[0].stride[i+1] * outputs[0].dim[i+1];
  // Set SplitInfo
  for (int i = 0; i < ndim; i++)
    outputs[0].split[i] = SplitInfo::NO_SPLIT;
  outputs[0].idx = 0;
}

Constant::~Constant(void)
{}

bool Constant::get_parameter(PMParameter para, int* value)
{
  return OpBase::get_parameter(para, value);
}

void Constant::collect_costs(float& exe_time, float& flops,
                             float& mem_acc, int& num_kernels)
{
  // TODO; implement
  assert(false);
}

ConstantKey::ConstantKey(int ndim, int* dims, OpType type)
{
  int idx = 0;
  keys[idx++] = ndim;
  for (int i = 0; i < ndim; i++)
    keys[idx++] = dims[i];
  keys[idx++] = type;
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(KEY_LENGTH == idx);
}

