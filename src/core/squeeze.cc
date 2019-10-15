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

TensorHandle Graph::squeeze(const TensorHandle input,
                            const std::vector<int>& axes)
{
  Op op = model->get_or_create_squeeze(*input, axes);
  add_edge(input->op, op, input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_squeeze(const Tensor& input,
                                const std::vector<int>& axes)
{
  // key is (input, axes)
  SqueezeKey key(input, axes);
  Squeeze* squeezeOp;
  if (squeeze.find(key) != squeeze.end()) {
    squeezeOp = squeeze[key];
  } else {
    squeezeOp = new Squeeze(this, input, axes);
    measure_squeeze_cost(squeezeOp);
    squeeze[key] = squeezeOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = squeezeOp;
  return ret;
}

Squeeze::Squeeze(Model* _model, const Tensor& _input,
                 const std::vector<int>& _axes)
: OpBase(_input, _model, OP_SQUEEZE), axes(_axes)
{
  numOutputs = 1;
  int n = 0;
  for (int i = 0; i < _input.numDim; i++) {
    bool squeezed = false;
    for (size_t idx = 0; idx < axes.size(); idx++)
      if (i == axes[idx])
        squeezed = true;
    if (!squeezed) {
      outputs[0].dim[n] = _input.dim[i];
      outputs[0].stride[n] = _input.stride[i];
      outputs[0].split[n] = _input.split[i];
      n++;
    } else {
      assert(_input.dim[i] == 1);
    }
  }
  outputs[0].numDim = n;
  outputs[0].idx = 0;
}

Squeeze::~Squeeze(void)
{}

bool Squeeze::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void Squeeze::collect_costs(float& exe_time, float& flops,
                            float& meme_acc, int& num_kernels)
{
  exe_time += runtime;
  num_kernels += 1;
  printf("        cost[Squeeze]: cost(%.4lf) total_cost(%.4lf)\n",
         runtime, exe_time);
}

SqueezeKey::SqueezeKey(const Tensor& input,
                      const std::vector<int>& axes)
{
  int idx = 0;
  for (size_t i = 0; i < axes.size(); i++)
    keys[idx++] = axes[i];
  input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}
