/* Copyright 2018 Stanford
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

void Graph::split(Tensor _input, int axis, int _num,
                  const int* _sizes, Tensor* outputs)
{
  int n = _num, sizes[MAX_NUM_OUTPUTS];
  for (int i = 0; i < n; i++)
    sizes[i] = _sizes[i];
  Op op = model->get_or_create_split(_input, axis, n, sizes);
  add_edge(_input.op, op, _input.idx, 0);
  for (int i = 0; i < n; i++) {
    outputs[i] = op.ptr->outputs[i];
    outputs[i].op = op;
  }
}

/*
void Graph::split(Tensor _input, int axis, int _num, Tensor* outputs)
{
  int sizes[MAX_NUM_OUTPUTS];
  SplitInfo parent = _input.split[axis], left, right;
  int curPos, oldPos = _input.dim[axis];
  for (int i = _num - 1; i >= 0; i--) {
    parent.divide(left, right, curPos);
    sizes[i] = oldPos - curPos;
    oldPos = curPos;
    parent = left;
  }
  Graph::split(_input, axis, _num, sizes, outputs);
}
*/

void Graph::split(Tensor _input, int axis, int size1, int size2, Tensor* outputs)
{
  int sizes[2];
  sizes[0] = size1;
  sizes[1] = size2;
  Graph::split(_input, axis, 2, sizes, outputs);
}

Op Model::get_or_create_split(Tensor _input, int axis, int n, int* sizes)
{
  // key ordering is:
  // axis, n, inputs[0].dim[0], ..., inputs[0].dim[axis-1],
  // inputs[0].dim[axis+1], ..., inputs[0].dim[nDims - 1]
  // sizes[0], ..., sizes[n-1]
  SplitKey key(_input, axis, n, sizes);
  Split* splitOp;
  if (split.find(key) != split.end()) {
    splitOp = split[key];
  } else {
    splitOp = new Split(this, _input, axis, n, sizes);
    measure_split_cost(splitOp);
    split[key] = splitOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = splitOp;
  return ret;
}

Op Model::get_or_create_split(Tensor _input, int axis, int n)
{
  int sizes[MAX_NUM_OUTPUTS];
  SplitInfo parent = _input.split[axis], left, right;
  int curPos, oldPos = _input.dim[axis];
  for (int i = n - 1; i > 0; i--) {
    parent.divide(left, right, curPos);
    sizes[i] = oldPos - curPos;
    oldPos = curPos;
    parent = left;
  }
  sizes[0] = oldPos;
  Op ret = get_or_create_split(_input, axis, n, sizes);
  return ret;
}

Split::Split(Model* _model, Tensor _input, int _axis, int n, int* _sizes)
  : OpBase(_input, model, OP_SPLIT), axis(_axis)
{
  assert(n <= MAX_NUM_OUTPUTS);
  numOutputs = n;
  for (int i = 0; i < n; i++)
    sizes[i] = _sizes[i];
  SplitInfo parent = inputs[0].split[axis], left, right;
  int oldPos = inputs[0].dim[axis], curPos;
  bool misMatch = false;
  for (int i = n - 1; i >= 0; i--) {
    outputs[i].numDim = inputs[0].numDim;
    for (int j = 0; j < inputs[0].numDim; j++)
      if (j != axis) {
        outputs[i].dim[j] = inputs[0].dim[j];
        outputs[i].stride[j] = inputs[0].stride[j];
        outputs[i].split[j] = inputs[0].split[j];
      } else {
        outputs[i].dim[j] = _sizes[i];
        outputs[i].stride[j] = inputs[0].stride[j];
        if (i > 0) {
          parent.divide(left, right, curPos);
        } else {
          curPos = 0;
          right = parent;
        }
        if (oldPos - curPos == _sizes[i])
          outputs[i].split[j] = right;
        else {
          misMatch = true;
          outputs[i].split[j] = SplitInfo::NO_SPLIT;
        }
        oldPos = curPos;
        parent = left;
      }
  }
  if (misMatch) {
    // Clear split info if mismatch
    for (int i = n - 1; i >= 0; i--)
      outputs[i].split[axis] = SplitInfo::NO_SPLIT;
  }
}

Split::~Split(void)
{}

bool Split::get_parameter(PMParameter para, int* value)
{
  switch (para) {
    case PM_AXIS:
      *value = axis;
      return true;
    default:
      return OpBase::get_parameter(para, value);
  }
}

void Split::map(void)
{
  size_t offset = 0;
  for (int i = 0; i < numOutputs; i++) {
    outputs[i].data_ptr = (DATATYPE*)inputs[0].data_ptr + offset;
    offset += outputs[i].dim[axis] * inputs[0].stride[axis];
  }
}

void Split::unmap(void)
{}

void Split::forward(bool block)
{}

void Split::collect_costs(float& exe_time, float& flops,
                          float& mem_acc, int& num_kernels)
{
  // cost metrics
  exe_time += 0;
  flops += 0;
  mem_acc += 0;
  num_kernels += 0;
  printf("        cost[Split]: numOutputs(%d) cost(%.4lf) total_cost(%.4lf)\n",
         numOutputs, 0.0f, exe_time);
}

void Model::measure_split_cost(Split* split)
{
  // We assume split cost is zero
  split->runtime = 0;
  if (print_cost)
    printf("        measure[split]: cost(%.4lf)\n", split->runtime);
}

// key ordering is:
// axis, n, sizes[0], ..., sizes[n-1], input
SplitKey::SplitKey(Tensor input, int axis, int n, int* sizes)
{
  int idx = 0;
  keys[idx++] = axis;
  keys[idx++] = n;
  for (int i = 0; i < n; i++)
    keys[idx++] = sizes[i];
  input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}
