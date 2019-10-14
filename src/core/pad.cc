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

TensorHandle Graph::pad(const TensorHandle _input,
                        const std::vector<int>& _pad_before,
                        const std::vector<int>& _pad_after,
                        float _pad_value)
{
  assert(_pad_before.size() == (size_t)_input.numDim);
  assert(_pad_after.size() == (size_t)_input.numDim);
  Op op = model->get_or_create_pad(*_input, _pad_before, _pad_after, _pad_value);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_pad(const Tensor& _input,
                            const std::vector<int>& _pad_before,
                            const std::vector<int>& _pad_after,
                            float _pad_value)
{
  PadKey key(_input, _pad_before, _pad_after, _pad_value);
  Pad* padOp;
  if (pad.find(key) != pad.end()) {
    padOp = pad[key];
  } else {
    padOp = new Pad(this, _input, _pad_before, _pad_after, _pad_value);
    measure_pad_cost(padOp);
    pad[key] = padOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = padOp;
  return ret;
}

Pad::Pad(Model* _model, const Tensor& _input,
         const std::vector<int>& _pad_before,
         const std::vector<int>& _pad_after,
         float _pad_value)
: OpBase(_input, _model, OP_PAD), pad_before(_pad_before),
pad_after(_pad_after), pad_value(_pad_value)
{
  numOutputs = 1;
  // Pad currently only support the defacult layout
  assert(_input.default_layout());
  outputs[0].numDim = _input.numDim;
  itn cnt = 1;
  for (int i = _input.numDim-1; i >= 0; i--) {
    outputs[0].dim[i] = _input.dim[i] + pad_before[i] + pad_after[i];
    outputs[0].stride[i] = cnt;
    outputs[0].split[i] = SplitInfo::NO_SPLIT;
    cnt *= outputs[0].dim[i];
  }
  outputs[0].idx = 0;
}

Pad::~Pad(void)
{
}

bool Pad::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void Reduce::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  exe_time += runtime;
  flops += inputs[0].volume();
  mem_acc += inputs[0].volume() + outputs[0].volume();
  num_kernels += 1;
  printf("      cost[Pad]: cost(%.4lf) total_cost(%.4lf)\n",
         runtime, exe_time);
}

PadKey::PadKey(const Tensor& _input,
               const std::vector<int>& _pad_before,
               const std::vector<int>& _pad_after,
               float _pad_value)
{
  //TODO: currently we do not include pad_value in the hash
  int idx = 0;
  keys[idx++] = _pad_before.size();
  for (size_t j = 0; j < _pad_before.size(); j++)
    keys[idx++] = _pad_before[j];
  for (size_t j = 0; j < _pad_after.size(); j++)
    keys[idx++] = _pad_after[j];
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}

