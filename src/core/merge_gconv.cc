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

#include "xflow/ops.h"
using namespace XFlow;

// Merge multiple group convs to a single group conv
TensorHandle Graph::merge_gconv(const TensorHandle _weight,
                                int count)
{
  // Currently the weight being merged must be 4D:
  // Count, Cin, KernelH, KernelW
  assert(_weight->numDim == 4);
  Op op = model->get_or_create_merge_gconv(*_weight, count);
  assert(op != Op::INVALID_OP);
  add_edge(_weight->op, op, _weight->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_merge_gconv(const Tensor& _weight,
                                    int count)
{
  // Check 1: weight must have 4D
  if (_weight.numDim != 4)
    return Op::INVALID_OP;
  // new group number must be an integer
  //if (_input.dim[1] % (_weight.dim[1] * count) != 0)
    //return Op::INVALID_OP;
  MergeGConvKey key(_weight, count);
  MergeGConv* mergeOp;
  if (merge_gconv.find(key) != merge_gconv.end()) {
    mergeOp = merge_gconv[key];
  } else {
    mergeOp = new MergeGConv(this, _weight, count);
    mergeOp->runtime = 0.0f;
    merge_gconv[key] = mergeOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = mergeOp;
  return ret;
}

MergeGConv::MergeGConv(Model* _model,
                       const Tensor& _weight,
                       int _count)
: OpBase(_weight, _model, OP_MERGE_GCONV), count(_count)
{
  assert(_weight.numDim == 4);
  numOutputs = 1;
  outputs[0].numDim = _weight.numDim;
  outputs[0].dim[0] = _weight.dim[0];
  outputs[0].dim[1] = _weight.dim[1] * count;
  outputs[0].dim[2] = _weight.dim[2];
  outputs[0].dim[3] = _weight.dim[3];
  outputs[0].stride[3] = 1;
  outputs[0].stride[2] = outputs[0].stride[3] * outputs[0].dim[3];
  outputs[0].stride[1] = outputs[0].stride[2] * outputs[0].dim[2];
  outputs[0].stride[0] = outputs[0].stride[1] * outputs[0].dim[1];
  // Set SplitInfo
  outputs[0].split[0] = _weight.split[0];
  outputs[0].split[1] = SplitInfo::NO_SPLIT;
  outputs[0].split[2] = _weight.split[2];
  outputs[0].split[3] = _weight.split[3];
  outputs[0].idx = 0;
  // assume that group number is an integer
}

MergeGConv::~MergeGConv(void)
{}

bool MergeGConv::get_parameter(PMParameter para, int* value)
{
  switch (para) {
    case PM_MERGE_GCONV_COUNT:
      *value = count;
      return true;
    default:
      return OpBase::get_parameter(para, value);
  }
}

void MergeGConv::collect_costs(float& exe_time, float& flops,
                               float& mem_acc, int& num_kernels)
{
  int outputSize = outputs[0].volume();
  int inputSize = inputs[0].volume();
  exe_time += runtime;
  flops += outputSize;
  mem_acc += inputSize + outputSize;
  num_kernels += 1;
}

// keys are (count, _weight)
MergeGConvKey::MergeGConvKey(const Tensor& _weight,
                             int count)
{
  assert(_weight.numDim == 4);
  int idx = 0;
  keys[idx++] = count;
  _weight.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}
