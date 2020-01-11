/* Copyright 2020 Stanford
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

TensorHandle Graph::broadcast_add(const TensorHandle _data,
                                        const TensorHandle _bias)
{
  Op op = model->get_or_create_broadcast_add(*_data, *_bias);
  add_edge(_data->op, op, _data->idx, 0);
  add_edge(_bias->op, op, _bias->idx, 1);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_broadcast_add(const Tensor& _data, const Tensor& _bias)
{
  BroadcastAddKey key(_data);
  BroadcastAdd* newop;
  if (broadcast_add.find(key) != broadcast_add.end()) {
    newop = broadcast_add[key];
  } else {
    newop = new BroadcastAdd(this, _data, _bias);
    //Assign a zero cost since it can be preprocessed
    // measure_fuse_conv_batchnorm_cost(fuseOp);
    newop->runtime = 0.0f;
    broadcast_add[key] = newop;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = newop;
  return ret;
}

BroadcastAdd::BroadcastAdd(Model* _model, const Tensor& _data, const Tensor& _bias)
: OpBase(_data, _bias, _model, OP_BROADCAST_ADD)
{
  assert(_data.numDim == 4);
  assert(_bias.numDim == 1);
  numOutputs = 1;
  outputs[0] = _data;
  outputs[0].idx = 0;
}

BroadcastAdd::~BroadcastAdd(void)
{}

bool BroadcastAdd::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void BroadcastAdd::collect_costs(float& exe_time, float& flops,
                                      float& mem_acc, int& num_kernels)
{
  // cost metrics
  exe_time += runtime;
  flops += outputs[0].volume();
  mem_acc += outputs[0].volume() * 2;
  num_kernels += 1;
  printf("        cost[BroadcastAdd]: i(%d %d %d %d) cost(%.4lf) total_cost(%.4lf)\n",
          inputs[0].dim[0], inputs[0].dim[1], inputs[0].dim[2], inputs[0].dim[3],
          runtime, exe_time);
}

// key is (_conv_w)
BroadcastAddKey::BroadcastAddKey(const Tensor& _data)
{
  int idx = 0;
  _data.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(KEY_LENGTH == idx);
}
