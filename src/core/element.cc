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

bool Model::broadcastable(const Tensor& t1,
                          const Tensor& t2)
{
  int num_dim = min(t1.numDim, t2.numDim);
  for (int dim = 0; dim < num_dim; dim++) {
    if ((t1.dim[t1.numDim-1-dim] != 1)
      &&(t2.dim[t2.numDim-1-dim] != 1)
      &&(t1.dim[t1.numDim-1-dim] != t2.dim[t2.numDim-1-dim]))
    {
      return false;
    }
  }
  return true;
}

TensorHandle Graph::element(OpType type,
                            const TensorHandle t1,
                            const TensorHandle t2)
{
  if (!model->broadcastable(*t1, *t2)) {
    fprintf(stderr, "Error: inputs could not be broadcast together");
    assert(false);
    return NULL;
  }
  Op op = model->get_or_create_element(type, *t1, *t2);
  add_edge(t1->op, op, t1->idx, 0);
  add_edge(t2->op, op, t2->idx, 1);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_element(OpType type,
                                const Tensor& t1,
                                const Tensor& t2)
{
  if (!broadcastable(t1, t2)) {
    return Op::INVALID_OP;
  }
  // key is (inputN, inputC, inputH, inputW, type)
  ElementKey key(t1, t2, type);
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
                 const Tensor& _t1,
                 const Tensor& _t2)
: OpBase(_t1, _t2, _model, _type)
{
  numOutputs = 1;
  int num_dim = max(_t1.numDim, _t2.numDim);
  outputs[0].numDim = num_dim;
  int total = 1;
  for (int i = 0; i < num_dim; i++) {
    int t1_idx = _t1.numDim-1-i;
    int t2_idx = _t2.numDim-1-i;
    int out_idx = num_dim-1-i;
    int dim1 = 1, dim2 = 1;
    if (t1_idx >= 0)
      dim1 = _t1.dim[t1_idx];
    if (t2_idx >= 0)
      dim2 = _t2.dim[t2_idx];
    outputs[0].dim[out_idx] = max(dim1, dim2);
    outputs[0].stride[out_idx] = total;
    total *= outputs[0].dim[out_idx];
    outputs[0].split[out_idx] = SplitInfo::NO_SPLIT;
    if (t1_idx >= 0 && _t1.dim[t1_idx] > 1) {
      outputs[0].split[out_idx] = _t1.split[t1_idx];
      if (t2_idx >= 0 && _t2.dim[t2_idx] > 1)
        otuputs[0].split[out_idx].combine(_t2.split[t2_idx]);
    } else if (t2_idx >= 0 && _t2.dim[t2_idx] > 1) {
      outputs[0].split[out_idx] = _t2.split[t2_idx];
    }
  }
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
ElementKey::ElementKey(const Tensor& t1,
                       const Tensor& t2,
                       OpType type)
{
  int idx = 0;
  keys[idx++] = type;
  t1.serialize(keys, idx);
  t2.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}
