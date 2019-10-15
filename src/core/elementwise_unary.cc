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

TensorHandle Graph::ceil(const TensorHandle _input)
{
  return elementwise_unary(_input, OP_CEIL);
}

TensorHandle Graph::exp(const TensorHandle _input)
{
  return elementwise_unary(_input, OP_EXP);
}

TensorHandle Graph::round(const TensorHandle _input)
{
  return elementwise_unary(_input, OP_ROUND);
}

TensorHandle Graph::log(const TensorHandle _input)
{
  return elementwise_unary(_input, OP_LOG);
}

TensorHandle Graph::logical_not(const TensorHandle _input)
{
  return elementwise_unary(_input, OP_LOGICAL_NOT);
}

TensorHandle Graph::sqrt(const TensorHandle _input)
{
  return elementwise_unary(_input, OP_SQRT);
}

TensorHandle Graph::elementwise_unary(const TensorHandle _input,
                                      OpType _type)
{
  Op op = model->get_or_create_elementwise_unary(*_input, _type);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_elementwise_unary(const Tensor& _input, OpType _type)
{
  ElementWiseUnaryKey key(_input, _type);
  ElementWiseUnary* unaryOp;
  if (element_unary.find(key) != element_unary.end()) {
    unaryOp = element_unary[key];
  } else {
    unaryOp = new ElementWiseUnary(this, _input, _type);
    measure_elementwise_unary_cost(unaryOp);
    element_unary[key] = unaryOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = unaryOp;
  return ret;
}

ElementWiseUnary::ElementWiseUnary(Model* _model, const Tensor& _input,
                                   OpType _type)
: OpBase(_input, _model, _type)
{
  numOutputs = 1;
  outputs[0] = _input;
  outputs[0].idx = 0;
}

ElementWiseUnary::~ElementWiseUnary(void)
{}

bool ElementWiseUnary::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void ElementWiseUnary::collect_costs(float& exe_time, float& flops,
                                     float& mem_acc, int& num_kernels)
{
  // cost metrics
  exe_time += runtime;
  flops += outputs[0].volume();
  mem_acc += inputs[0].volume();
  num_kernels += 1;
  printf("        cost[ElementWiseUnary]: mode(%d) cost(%.4lf) total_cost(%.4lf)\n",
         type, runtime, exe_time);
}

ElementWiseUnaryKey::ElementWiseUnaryKey(const Tensor& _input, OpType _type)
{
  int idx = 0;
  keys[idx++] = _type;
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}
