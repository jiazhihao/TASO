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

TensorHandle Graph::element(OpType type,
                            const TensorHandle t1,
                            const TensorHandle t2)
{
  assert(t1->numDim == t2->numDim);
  for (int i = 0; i < t1->numDim; i++)
    assert(t1->dim[i] == t2->dim[i]);
  Op op = model->get_or_create_element(type, *t1, *t2);
  add_edge(t1->op, op, t1->idx, 0);
  add_edge(t2->op, op, t2->idx, 1);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_element(OpType type,
                                Tensor t1, Tensor t2)
{
  // key is (inputN, inputC, inputH, inputW, type)
  ElementKey key(t1, type);
  Element* eleOp;
  if (element.find(key) != element.end()) {
    eleOp = element[key];
  } else {
    eleOp = new Element(this, type, t1, t2);
    measure_element_cost(eleOp);
    element[key] = eleOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = eleOp;
  return ret;
}

Element::Element(Model* _model, OpType _type,
                 Tensor _t1, Tensor _t2)
: OpBase(_t1, _t2, _model, _type)
{
  numOutputs = 1;
  outputs[0] = _t1;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputs[0].split[i].combine(_t2.split[i]);
  outputs[0].idx = 0;
}

Element::~Element(void)
{}

bool Element::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void Element::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  int outputSize = 1, inputSize = 1;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  for (int i = 0; i < inputs[0].numDim; i++)
    inputSize *= inputs[0].dim[i];
  // cost metrics
  exe_time += runtime;
  flops += outputSize;
  mem_acc += inputSize * 2;
  num_kernels += 1;
  printf("        cost[Element]: cost(%.4lf) total_cost(%.4lf)\n", runtime, exe_time);
}

// Key ordering: type, input
ElementKey::ElementKey(Tensor input, OpType type)
{
  int idx = 0;
  keys[idx++] = type;
  input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}
