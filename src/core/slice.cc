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

TensorHandle Graph::slice(const TensorHandle _input,
                          const std::vector<int>& _start,
                          const std::vector<int>& _end,
                          const std::vector<int>& _axes,
                          const std::vector<int>& _steps)
{
  Op op = model->get_or_create_slice(*_input, _start, _end, _axes, _steps);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_slice(const Tensor& _input,
                              const std::vector<int>& _start,
                              const std::vector<int>& _end,
                              const std::vector<int>& _axes,
                              const std::vector<int>& _steps)
{
  if (_start.size() != _end.size())
    return Op::INVALID_OP;
  if (_start.size() != _axes.size())
    return Op::INVALID_OP;
  if (_start.size() != _steps.size())
    return Op::INVALID_OP;
  SliceKey key(_input, _start, _end, _axes, _steps);
  Slice* sliceOp;
  if (slice.find(key) != slice.end()) {
    sliceOp = slice[key];
  } else {
    sliceOp = new Slice(this, _input, _start, _end, _axes, _steps);
    measure_slice_cost(sliceOp);
    slice[key] = sliceOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = sliceOp;
  return ret;
}

Slice::Slice(Model* _model, const Tensor& _input,
             const std::vector<int>& _start,
             const std::vector<int>& _end,
             const std::vector<int>& _axes,
             const std::vector<int>& _steps)
: OpBase(_input, _model, OP_SLICE),
  start(_start), end(_end), axes(_axes), steps(_steps)
{
  assert(_start.size() == _end.size());
  assert(_start.size() == _axes.size());
  assert(_start.size() == _steps.size());
  numOutputs = 1;
  outputs[0].numDim = _input.numDim;
  // Currently assume the input tensor has the default layout
  assert(_input.default_layout());
  int total = 1;
  for (int i = _input.numDim-1; i >= 0; i--) {
    int idx = -1;
    for (size_t j = 0; j < _start.size(); j++)
      if (_axes[j] == i) {
        idx = j;
        break;
      }
    if (idx == -1) {
      outputs[0].dim[i] = _input.dim[i];
      outputs[0].split[i] = _input.split[i];
      outputs[0].stride[i] = total;
      total *= outputs[0].dim[i];
    } else {
      int start_pos = min(_start[idx], _input.dim[i]);
      int end_pos = min(_end[idx], _input.dim[i]);
      int dim_size = (end_pos - start_pos) / _steps[idx] + 1;
      outputs[0].dim[i] = dim_size;
      outputs[0].split[i] = SplitInfo::NO_SPLIT;
      outputs[0].stride[i] = total;
      total *= outputs[0].dim[i];
    }
  }
  outputs[0].idx = 0;
}

Slice::~Slice(void)
{}

bool Slice::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void Slice::collect_costs(float& exe_time, float& flops,
                          float& mem_acc, int& num_kernels)
{
  exe_time += runtime;
}

SliceKey::SliceKey(const Tensor& _input,
                   const std::vector<int>& _start,
                   const std::vector<int>& _end,
                   const std::vector<int>& _axes,
                   const std::vector<int>& _steps)
{
  int idx = 0;
  keys[idx++] = _start.size();
  for (size_t i = 0; i < _start.size(); i++)
    keys[idx++] = _start[i];
  for (size_t i = 0; i < _end.size(); i++)
    keys[idx++] = _end[i];
  for (size_t i = 0; i < _axes.size(); i++)
    keys[idx++] = _axes[i];
  for (size_t i = 0; i < _steps.size(); i++)
    keys[idx++] = _steps[i];
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}
