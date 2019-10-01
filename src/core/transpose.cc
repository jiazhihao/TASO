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

int permutation_to_index(const std::vector<int>& perm)
{
  // check perm
  for (size_t i = 0; i < perm.size(); i++) {
    assert(perm[i] >= 0 && perm[i] < perm.size());
    for (size_t j = i + 1; j < perm.size(); j++)
      assert(perm[i] != perm[j]);
  }
  int idx = 0;
  for (size_t i = 0; i < perm.size(); i++)
    idx = idx * perm.size() + perm[i];
  return idx;
}

TensorHandle Graph::transpose(const TensorHandle _input,
                              const std::vector<int>& perm,
                              bool _shuffle)
{
  Op op = model->get_or_create_transpose(*_input, perm, _shuffle);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_transpose(Tensor _input, int permIdx,
                                  bool _shuffle)
{
  int ndim = _input.numDim;
  std::vector<int> permVec;
  int permArray[MAX_DIM];
  for (int i = ndim - 1; i >= 0; i--) {
    permArray[i] = permIdx % ndim;
    permIdx = permIdx / ndim;
  }
  if (permIdx != 0) {
    return Op::INVALID_OP;
  }
  for (int i = 0; i < ndim; i++)
    for (int j = i + 1; j < ndim; j++)
      if (permArray[i] != permArray[j]) {
        return Op::INVALID_OP;
      }
  for (int i = 0; i < ndim; i++)
    permVec.push_back(permArray[i]);
  return get_or_create_transpose(_input, permVec, _shuffle);
}

Op Model::get_or_create_transpose(Tensor _input,
                                  const std::vector<int>& perm,
                                  bool _shuffle)
{
  TransposeKey key(_input, perm, _shuffle);
  Transpose* transposeOp;
  if (transpose.find(key) != transpose.end()) {
    transposeOp = transpose[key];
  } else {
    transposeOp = new Transpose(this, _input, perm, _shuffle);
    measure_transpose_cost(transposeOp);
    transpose[key] = transposeOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = transposeOp;
  return ret;
}

Transpose::Transpose(Model* _model, Tensor _input,
                     const std::vector<int>& _perm,
                     bool _shuffle)
: OpBase(_input, _model, OP_TRANSPOSE), shuffle(_shuffle)
{
  assert(shuffle);
  permIdx = permutation_to_index(_perm);
  assert(_input.numDim == _perm.size());
  numOutputs = 1;
  // set dims and strides
  outputs[0].numDim = _input.numDim;
  for (int i = 0; i < _perm.size(); i++) {
    outputs[0].dim[i] = _input.dim[_perm[i]];
    outputs[0].split[i] = _input.split[_perm[i]];
  }
  if (shuffle) {
    int size = 1;
    for (int i = _perm.size() - 1; i >= 0; i--) {
      outputs[0].stride[i] = size;
      size *= outputs[0].dim[i];
    }
    assert(size == outputs[0].volume());
  } else {
    for (int i = 0; i < _perm.size(); i++)
      outputs[0].stride[i] = _input.stride[_perm[i]];
  }
  outputs[0].idx = 0;
}

Transpose::~Transpose(void)
{}

bool Transpose::get_parameter(PMParameter para, int* value)
{
  switch (para) {
    case PM_NUMDIM:
      *value = outputs[0].numDim;
      return true;
    case PM_PERM:
      *value = permIdx;
      return true;
    case PM_OUTSHUFFLE:
      *value = (int) shuffle;
      return true;
    default:
      return OpBase::get_parameter(para, value);
  }
}

void Transpose::collect_costs(float& exe_time, float& flops,
                              float& mem_acc, int& num_kernels)
{
  if (shuffle) {
    exe_time += runtime;
    flops += outputs[0].volume();
    mem_acc += outputs[0].volume();
    num_kernels += 1;
  }
}

TransposeKey::TransposeKey(Tensor _input,
                           const std::vector<int>& perm,
                           bool _shuffle)
{
  int idx = 0;
  keys[idx++] = permutation_to_index(perm);
  keys[idx++] = (int) _shuffle;
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}

