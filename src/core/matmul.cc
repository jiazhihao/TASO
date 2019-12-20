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

TensorHandle Graph::fc(const TensorHandle _input,
                       int _outputC,
                       ActiMode acti)
{
  assert(_input->numDim == 2);
  const int dims[2] = {_outputC, _input->dim[1]};
  int total = dims[0] * dims[1];
  // Randomly initialize weights
  DATATYPE* data = (DATATYPE*) malloc(total * sizeof(DATATYPE));
  for (int i = 0; i < total; i++)
    data[i] = (DATATYPE)std::rand() / RAND_MAX;
  TensorHandle weight = new_weight(2, dims, data);
  free(data);
  return matmul(_input, weight, acti);
}

TensorHandle Graph::matmul(const TensorHandle _input,
                           const TensorHandle _weight,
                           ActiMode acti)
{
  Op op = model->get_or_create_matmul(*_input, *_weight, acti);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  add_edge(_weight->op, op, _weight->idx, 1);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_matmul(Tensor _input, Tensor _weight,
                               ActiMode _acti)
{
  if (_input.numDim != _weight.numDim)
    return Op::INVALID_OP;
  for (int i = 0; i < _input.numDim - 2; i++)
    if (_input.dim[i] != _weight.dim[i])
      return Op::INVALID_OP;
  if (_input.dim[_input.numDim-1] != _weight.dim[_weight.numDim-2])
    return Op::INVALID_OP;
  // key is (inputX, inputN, inputC, outputC, acti)
  MatmulKey key(_input, _weight, _acti);
  Matmul* matmulOp;
  if (matmul.find(key) != matmul.end()) {
    matmulOp = matmul[key];
  } else {
    matmulOp = new Matmul(this, _input, _weight, _acti);
    measure_matmul_cost(matmulOp);
    matmul[key] = matmulOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = matmulOp;
  return ret;
}

Matmul::Matmul(Model* _model, Tensor _input, Tensor _weight, ActiMode _activation)
: OpBase(_input, _weight, _model, OP_MATMUL), activation(_activation)
{
  int numDim = _input.numDim;
  assert(numDim == _weight.numDim);
  for (int i = 0; i < numDim - 2; i++)
    assert(_input.dim[i] == _weight.dim[i]);
  assert(_input.dim[numDim-1] == _weight.dim[numDim-2]);
  numOutputs = 1;
  // set dims and strides
  outputs[0].numDim = numDim;
  for (int i = 0; i < numDim-1; i++)
    outputs[0].dim[i] = _input.dim[i];
  outputs[0].dim[numDim-1] = _weight.dim[numDim-1];
  set_layout();
  // set SplitInfo
  for (int i = 0; i < numDim-2; i++) {
    if (_input.split[i] == _weight.split[i])
      outputs[0].split[i] = _input.split[i];
    else
      outputs[0].split[i] = SplitInfo::NO_SPLIT;
  }
  outputs[0].split[numDim-2] = _input.split[numDim-2];
  outputs[0].split[numDim-1] = _weight.split[numDim-1];
  outputs[0].idx = 0;
}

Matmul::~Matmul(void)
{}

bool Matmul::get_int_parameter(PMParameter para, int* value)
{
  switch (para) {
    case PM_ACTI:
      *value = (int) activation;
      return true;
    default:
      return OpBase::get_int_parameter(para, value);
  }
}

void Matmul::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  int outputSize = 1, inputSize = 1;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  for (int i = 0; i < inputs[0].numDim; i++)
    inputSize *= inputs[0].dim[i];
  // cost metrics
  exe_time += runtime;
  assert(inputs[0].numDim == inputs[1].numDim);
  flops += outputSize * inputs[0].dim[inputs[0].numDim-1];
  mem_acc += inputSize;
  num_kernels += 1;
  printf("        cost[Matmul]: %s %s cost(%.4lf) total_cost(%.4lf)\n",
         inputs[0].to_string("input").c_str(),
         inputs[1].to_string("weight").c_str(),
         runtime, exe_time);
}

// key is (inputN, inputC, outputC, acti)
MatmulKey::MatmulKey(Tensor _input, Tensor _weight, ActiMode _mode)
{
  assert(_input.numDim == _weight.numDim);
  int idx = 0;
  keys[idx++] = (int)(_mode);
  _input.serialize(keys, idx);
  _weight.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}

