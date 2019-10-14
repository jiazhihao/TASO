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

TensorHandle Graph::reduce_argmax(const TensorHandle _input,
                                  const std::vector<int>& axes,
                                  bool keepdims)
{
  return reduce(_input, OP_REDUCE_ARGMAX, axes, keepdims);
}

TensorHandle Graph::reduce_argmin(const TensorHandle _input,
                                  const std::vector<int>& axes,
                                  bool keepdims)
{
  return reduce(_input, OP_REDUCE_ARGMIN, axes, keepdims);
}

TensorHandle Graph::reduce_max(const TensorHandle _input,
                               const std::vector<int>& axes,
                               bool keepdims)
{
  return reduce(_input, OP_REDUCE_MAX, axes, keepdims);
}

TensorHandle Graph::reduce_mean(const TensorHandle _input,
                                const std::vector<int>& axes,
                                bool keepdims)
{
  return reduce(_input, OP_REDUCE_MEAN, axes, keepdims);
}

TensorHandle Graph::reduce_min(const TensorHandle _input,
                               const std::vector<int>& axes,
                               bool keepdims)
{
  return reduce(_input, OP_REDUCE_MIN, axes, keepdims);
}

TensorHandle Graph::reduce_prod(const TensorHandle _input,
                                const std::vector<int>& axes,
                                bool keepdims)
{
  return reduce(_input, OP_REDUCE_PROD, axes, keepdims);
}

TensorHandle Graph::reduce_sum(const TensorHandle _input,
                               const std::vector<int>& axes,
                               bool keepdims)
{
  return reduce(_input, OP_REDUCE_SUM, axes, keepdims);
}

TensorHandle Graph::reduce(const TensorHandle _input,
                           OpType _type,
                           const std::vector<int>& axes,
                           bool keepdims)
{
  for (size_t i = 0; i < axes.size(); i++) {
    assert(axes[i] >= 0);
    assert(axes[i] < _input->numDim);
  }
  Op op = model->get_or_create_reduce(*_input, _type, axes, keepdims);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_reduce(const Tensor& _input,
                               OpType _type,
                               const std::vector<int>& axes,
                               bool keepdims)
{
  ReduceKey key(_input, _type, axes, keepdims);
  Reduce* reduceOp;
  if (reduce.find(key) != reduce.end()) {
    reduceOp = reduce[key];
  } else {
    reduceOp = new Reduce(this, _input, _type, axes, keepdims);
    measure_reduce_cost(reduceOp);
    reduce[key] = reduceOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = reduceOp;
  return ret;
}

Reduce::Reduce(Model* _model, const Tensor& _input, OpType _type,
               const std::vector<int>& _axes, bool _keepdims)
: OpBase(_input, _model, _type), keepdims(_keepdims), axes(_axes)
{
  numOutputs = 1;
  // Reduce currently only support the defacult layout
  assert(_input.default_layout());
  if (keepdims) {
    outputs[0].numDim = _input.numDim;
    int cnt = 1;
    for (int i = outputs[0].numDim-1; i >= 0; i--) {
      bool reduced = false;
      for (size_t j = 0; j < axes.size(); j++)
        if (axes[j] == i)
          reduced = true;
      outputs[0].stride[i] = cnt;
      if (reduced) {
        outputs[0].dim[i] = 1;
        outputs[0].split[i] = SplitInfo::NO_SPLIT;
      } else {
        outputs[0].dim[i] = _input.dim[i];
        outputs[0].split[i] = _input.split[i];
      }
      cnt = cnt * outputs[0].dim[i];
    }
  } else {
    outputs[0].numDim = _input.numDim-axes.size();
    int cnt = 1, dim = outputs[0].numDim-1;
    for (int i = _input.numDim-1; i >= 0; i--) {
      bool reduced = false;
      for (size_t j = 0; j < axes.size(); j++)
        if (axes[j] == i)
          reduced = true;
      if (reduced)
        continue;
      outputs[0].stride[dim] = cnt;
      outputs[0].dim[dim] = _input.dim[i];
      outputs[0].split[dim] = _input.split[i];
      cnt = cnt * outputs[0].dim[dim];
      dim = dim - 1;
    }
    // Cannot have duplicated reduce axes
    assert(dim == -1);
  }
  outputs[0].idx = 0;
}

Reduce::~Reduce(void)
{}

bool Reduce::get_int_parameter(PMParameter para, int* value)
{
  switch (para) {
    case PM_KEEP_DIMS:
      *value = keepdims;
      return true;
    default:
      return OpBase::get_int_parameter(para, value);
  }
}

void Reduce::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  exe_time += runtime;
  flops += inputs[0].volume();
  mem_acc += inputs[0].volume() + outputs[0].volume();
  num_kernels += 1;
  printf("      cost[Reduce]: cost(%.4lf) total_cost(%.4lf)\n",
         runtime, exe_time);
}

ReduceKey::ReduceKey(const Tensor& _input, OpType _type,
                     const std::vector<int>& axes, bool keepdims)
{
  int idx = 0;
  keys[idx++] = _type;
  keys[idx++] = (int)(keepdims);
  keys[idx++] = axes.size();
  for (size_t j = 0; j < axes.size(); j++)
    keys[idx++] = axes[j];
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}

