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

void Graph::topk(const TensorHandle _input,
                 int _axis, int _numk,
                 bool _largest, bool _sorted,
                 Tensor* outputs)
{
  Op op = model->get_or_create_topk(*_input, _numk, _axis, _largest, _sorted);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  outputs[0] = op.ptr->outputs[0];
  outputs[0].op = op;
  outputs[1] = op.ptr->outputs[1];
  outputs[1].op = op;
}

Op Model::get_or_create_topk(const Tensor& _input,
                             int _axis, int _numk,
                             bool _largest, bool _sorted)
{
  TopKKey key(_input, _axis, _numk, _largest, _sorted);
  TopK* topkOp;
  if (topk.find(key) != topk.end()) {
    topkOp = topk[key];
  } else {
    topkOp = new TopK(this, _input, _axis, _numk, _largest, _sorted);
    measure_topk_cost(topkOp);
    topk[key] = topkOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = topkOp;
  return ret;
}

TopK::TopK(Model* _model, const Tensor& _input,
           int _axis, int _numk, bool _largest, bool _sorted)
: OpBase(_input, _model, OP_TOPK), axis(_axis),
  largest(_largest), sorted(_sorted)
{
  numOutputs = 2;
  for (int i = 0; i < 2; i++) {
    outputs[i].numDim = _input.numDim;
    int total = 1;
    for (int j = _input.numDim-1; j >= 0; j--) {
      if (j != axis)
        outputs[i].dim[j] = _input.dim[j];
      else
        outputs[i].dim[j] = _numk;
      outputs[i].stride[j] = total;
      total *= outputs[i].dim[j];
      outputs[i].split[j] = SplitInfo::NO_SPLIT;
    }
    outputs[i].idx = i;
  }
}

TopK::~TopK(void)
{}

bool TopK::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void TopK::collect_costs(float& exe_time, float& flops,
                         float& mem_acc, int& num_kernels)
{
  exe_time += runtime;
  num_kernels += 1;
  printf("      cost[TopK]: cost(%.4lf) total_cost(%.4lf)\n",
         runtime, exe_time);
}

TopKKey::TopKKey(const Tensor& _input,
                 int _axis, int _numk,
                 bool _largest, bool _sorted)
{
  int idx = 0;
  keys[idx++] = _axis;
  keys[idx++] = _numk;
  keys[idx++] = (int)_largest;
  keys[idx++] = (int)_sorted;
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}

