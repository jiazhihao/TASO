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

TensorHandle Graph::unsqueeze(const TensorHandle input,
                              const std::vector<int>& axes)
{
  Op op = model->get_or_create_unsqueeze(*input, axes);
  add_edge(input->op, op, input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_unsqueeze(const Tensor& input,
                                  const std::vector<int>& axes)
{
  UnsqueezeKey key(input, axes);
  Unsqueeze* unsqzOp;
  if (unsqueeze.find(key) != unsqueeze.end()) {
    unsqzOp = unsqueeze[key];
  } else {
    unsqzOp = new Unsqueeze(this, input, axes);
    measure_unsqueeze_cost(unsqzOp);
    unsqueeze[key] = unsqzOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = unsqzOp;
  return ret;
}

Unsqueeze::Unsqueeze(Model* _model, const Tensor& _input,
                     const std::vector<int>& _axes)
: OpBase(_input, _model, OP_UNSQUEEZE), axes(_axes)
{
  numOutputs = 1;
  outputs[0].numDim = _input.numDim + axes.size();
  int n = 0;
  for (int i = 0; i < outputs[0].numDim; i++) {
    bool unsqueezed = false;
    for (size_t j = 0; j < axes.size(); j++)
      if (axes[j] == i)
        unsqueezed = true;
    if (unsqueezed) {
    } else {
      outputs[0].dim[i] = _input.dim[n];
      outputs[0].stride[i] = _input.stride[n];
      outputs[0].split[i] = _input.split[n];
      n ++;
    }
  }
  assert(n == _input.numDim);
  outputs[0].idx = 0;
}

Unsqueeze::~Unsqueeze(void)
{}

bool Unsqueeze::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void Unsqueeze::collect_costs(float& exe_time, float& flops,
                              float& mem_acc, int& num_kernels)
{
  exe_time += runtime;
  num_kernels += 1;
}

UnsqueezeKey::UnsqueezeKey(const Tensor& input, const std::vector<int>& axes)
{
  int idx = 0;
  for (size_t i = 0; i < axes.size(); i++)
    keys[idx++] = axes[i];
  input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}
