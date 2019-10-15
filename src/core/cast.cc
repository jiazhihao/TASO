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

TensorHandle Graph::cast(const TensorHandle _input, DataType _datatype)
{
  Op op = model->get_or_create_cast(*_input, _datatype);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_cast(const Tensor& _input, DataType _datatype)
{
  CastKey key(_input, _datatype);
  Cast* castOp;
  if (cast.find(key) != cast.end()) {
    castOp = cast[key];
  } else {
    castOp = new Cast(this, _input, _datatype);
    measure_cast_cost(castOp);
    cast[key] = castOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = castOp;
  return ret;
}

Cast::Cast(Model* _model, const Tensor& _input, DataType _datatype)
: OpBase(_input, _model, OP_CAST)
{
  numOutputs = 1;
  outputs[0] = _input;
  outputs[0].idx = 0;
}

Cast::~Cast(void)
{}

bool Cast::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void Cast::collect_costs(float& exe_time, float& flops,
                         float& mem_acc, int& num_kernels)
{
  // cost metrics
  exe_time += runtime;
  flops += outputs[0].volume();
  mem_acc += inputs[0].volume();
  num_kernels += 1;
  printf("        cost[Cast]: cost(%.4lf) total_cost(%.4lf)\n",
         runtime, exe_time);
}

CastKey::CastKey(const Tensor& _input, DataType _datatype)
{
  int idx = 0;
  keys[idx++] = _datatype;
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}
