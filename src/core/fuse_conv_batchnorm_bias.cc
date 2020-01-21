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

// Preproccessing weights to merge conv and batchnorm
TensorHandle Graph::fuse_conv_batchnorm_bias(const TensorHandle _scale,
                                        const TensorHandle _bias,
                                        const TensorHandle _mean,
                                        const TensorHandle _var)
{
  Op op = model->get_or_create_fuse_conv_batchnorm_bias(
      *_scale, *_bias, *_mean, *_var);
  add_edge(_scale->op, op, _scale->idx, 0);
  add_edge(_bias->op, op, _bias->idx, 1);
  add_edge(_mean->op, op, _mean->idx, 2);
  add_edge(_var->op, op, _var->idx, 3);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_fuse_conv_batchnorm_bias(const Tensor& _scale,
                                            const Tensor& _bias,
                                            const Tensor& _mean,
                                            const Tensor& _var)
{
  FuseConvBatchNormBiasKey key(_scale); // to do
  FuseConvBatchNormBias* fuseOp;
  if (fuse_conv_batchnorm_bias.find(key) != fuse_conv_batchnorm_bias.end()) {
    fuseOp = fuse_conv_batchnorm_bias[key];
  } else {
    fuseOp = new FuseConvBatchNormBias(this, _scale, _bias, _mean, _var);
    //Assign a zero cost since it can be preprocessed
    // measure_fuse_conv_batchnorm_cost(fuseOp);
    fuseOp->runtime = 0.0f;
    fuse_conv_batchnorm_bias[key] = fuseOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = fuseOp;
  return ret;
}

FuseConvBatchNormBias::FuseConvBatchNormBias(Model* _model,
                                     const Tensor& _scale,
                                     const Tensor& _bias,
                                     const Tensor& _mean,
                                     const Tensor& _var)
: OpBase(_scale, _bias, _mean, _var, _model, OP_FUSE_CONV_BATCHNORM_BIAS)
{
  assert(_scale.numDim == 1);
  numOutputs = 1;
  outputs[0] = _scale;
  outputs[0].idx = 0;
}

FuseConvBatchNormBias::~FuseConvBatchNormBias(void)
{}

bool FuseConvBatchNormBias::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void FuseConvBatchNormBias::collect_costs(float& exe_time, float& flops,
                                      float& mem_acc, int& num_kernels)
{
  // cost metrics
  exe_time += runtime;
  flops += outputs[0].volume();
  mem_acc += outputs[0].volume() * 2;
  num_kernels += 1;
  printf("        cost[FuseConvBatchNormBias]: i(%d) cost(%.4lf) total_cost(%.4lf)\n",
          inputs[0].dim[0],
          runtime, exe_time);
}

// key is (_conv_w)
FuseConvBatchNormBiasKey::FuseConvBatchNormBiasKey(const Tensor& _scale)
{
  int idx = 0;
  _scale.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(KEY_LENGTH == idx);
}
