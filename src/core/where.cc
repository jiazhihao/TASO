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

TensorHandle Graph::where(const TensorHandle _cond,
                          const TensorHandle _x,
                          const TensorHandle _y)
{
  if (!model->broadcastable(*_cond, *_x)) {
    fprintf(stderr, "Error: cond and x could not be broadcast together");
    assert(false);
    return NULL;
  }
  if (!model->broadcastable(*_cond, *_y)) {
    fprintf(stderr, "Error: cond and y could not be broadcast together");
    assert(false);
    return NULL;
  }
  if (!model->broadcastable(*_x, *_y)) {
    fprintf(stderr, "Error: x and y could not be broadcast together");
    assert(false);
    return NULL;
  }
  Op op = model->get_or_create_where(*_cond, *_x, *_y);
  assert(op != Op::INVALID_OP);
  add_edge(_cond->op, op, _cond->idx, 0);
  add_edge(_x->op, op, _x->idx, 1);
  add_edge(_y->op, op, _y->idx, 2);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_where(const Tensor& _cond,
                              const Tensor& _x,
                              const Tensor& _y)
{
  if (!broadcastable(_cond, _x)) {
    return Op::INVALID_OP;
  }
  if (!broadcastable(_cond, _y)) {
    return Op::INVALID_OP;
  }
  if (!broadcastable(_x, _y)) {
    return Op::INVALID_OP;
  }
  WhereKey key(_cond, _x, _y);
  Where* whereOp;
  if (where.find(key) != where.end()) {
    whereOp = where[key];
  } else {
    whereOp = new Where(this, _cond, _x, _y);
    measure_where_cost(whereOp);
    where[key] = whereOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = whereOp;
  return ret;
}

Where::Where(Model* _model, const Tensor& _cond,
             const Tensor& _x, const Tensor& _y)
: OpBase(_cond, _x, _y, _model, OP_WHERE)
{
  numOutputs = 1;
  assert(model->broadcastable(_cond, _x));
  assert(model->broadcastable(_cond, _y));
  assert(model->broadcastable(_x, _y));
  int num_dim = max(_cond.numDim, max(_x.numDim, _y.numDim));
  int total = 1;
  for (int i = 0; i < num_dim; i++) {
    int cond_idx = _cond.numDim-1-i;
    int x_idx = _x.numDim-1-i;
    int y_idx = _y.numDim-1-i;
    int out_idx = num_dim-1-i;
    int dim_size = 1;
    if (cond_idx >= 0)
      dim_size = max(dim_size, _cond.dim[cond_idx]);
    if (x_idx >= 0)
      dim_size = max(dim_size, _x.dim[x_idx]);
    if (y_idx >= 0)
      dim_size = max(dim_size, _y.dim[y_idx]);
    outputs[0].dim[out_idx] = dim_size;
    outputs[0].stride[out_idx] = total;
    total *= outputs[0].dim[out_idx];
    outputs[0].split[out_idx] = SplitInfo::NO_SPLIT;
    if (cond_idx >= 0 && _cond.dim[cond_idx] > 1) {
      outputs[0].split[out_idx] = _cond.split[cond_idx];
      if (x_idx >= 0 && _x.dim[x_idx] > 1)
        outputs[0].split[out_idx].combine(_x.split[x_idx]);
      if (y_idx >= 0 && _y.dim[y_idx] > 1)
        outputs[0].split[out_idx].combine(_y.split[y_idx]);
    } else if (x_idx >= 0 && _x.dim[x_idx] > 1) {
      outputs[0].split[out_idx] = _x.split[x_idx];
      if (y_idx >= 0 && _y.dim[y_idx] > 1)
        outputs[0].split[out_idx].combine(_y.split[y_idx]);
    } else if (y_idx >= 0 && _y.dim[y_idx] > 1) {
      outputs[0].split[out_idx] = _y.split[y_idx];
    }
  }
  outputs[0].idx = 0;
}

Where::~Where(void)
{}

bool Where::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void Where::collect_costs(float& exe_time, float& flops,
                          float& mem_acc, int& num_kernels)
{
  // cost metrics
  exe_time += runtime;
  flops += outputs[0].volume();
  mem_acc += 4 * outputs[0].volume();
  num_kernels += 1;
  printf("        cost[Where]: cost(%.4lf) total_cost(%.4lf)\n", runtime, exe_time);
}

WhereKey::WhereKey(const Tensor& _cond,
                   const Tensor& _x,
                   const Tensor& _y)
{
  int idx = 0;
  _cond.serialize(keys, idx);
  _x.serialize(keys, idx);
  _y.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}
