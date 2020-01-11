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

// // Preproccessing weights to merge conv and batchnorm
// TensorHandle Graph::fuse_conv_batchnorm_alpha_var(const TensorHandle _conv_w,
//                                         const TensorHandle _scale,
//                                         const TensorHandle _var)
// {
//   Op op = model->get_or_create_fuse_conv_batchnorm_alpha_var(
//       *_conv_w, *_scale, *_var);
//   add_edge(_conv_w->op, op, _conv_w->idx, 0);
//   add_edge(_scale->op, op, _scale->idx, 1);
//   add_edge(_var->op, op, _var->idx, 2);
//   TensorHandle t = new Tensor(op.ptr->outputs[0]);
//   t->op = op;
//   return t;
// }

Op Model::get_or_create_fuse_conv_batchnorm_alpha_var(const Tensor& _conv_w,
                                            const Tensor& _scale,
                                            const Tensor& _var)
{
  FuseConvBatchNormAlphaVarKey key(_conv_w);
  FuseConvBatchNormAlphaVar* fuseOp;
  if (fuse_conv_batchnorm_alpha_var.find(key) != fuse_conv_batchnorm_alpha_var.end()) {
    fuseOp = fuse_conv_batchnorm_alpha_var[key];
  } else {
    fuseOp = new FuseConvBatchNormAlphaVar(this, _conv_w, _scale, _var);
    //Assign a zero cost since it can be preprocessed
    // measure_fuse_conv_batchnorm_cost(fuseOp);
    fuseOp->runtime = 0.0f;
    fuse_conv_batchnorm_alpha_var[key] = fuseOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = fuseOp;
  return ret;
}

FuseConvBatchNormAlphaVar::FuseConvBatchNormAlphaVar(Model* _model,
                                     const Tensor& _conv_w,
                                     const Tensor& _scale,
                                     const Tensor& _var)
: OpBase(_conv_w, _scale, _var, _model, OP_FUSE_CONV_BATCHNORM_ALPHA_VAR)
{
  assert(_conv_w.numDim == 4);
  numOutputs = 1;
  outputs[0] = _conv_w;
  outputs[0].idx = 0;
}

FuseConvBatchNormAlphaVar::~FuseConvBatchNormAlphaVar(void)
{}

bool FuseConvBatchNormAlphaVar::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void FuseConvBatchNormAlphaVar::collect_costs(float& exe_time, float& flops,
                                      float& mem_acc, int& num_kernels)
{
  // cost metrics
  exe_time += runtime;
  flops += outputs[0].volume();
  mem_acc += outputs[0].volume() * 2;
  num_kernels += 1;
  printf("        cost[FuseConvBatchNormAlphaVar]: i(%d %d %d %d) cost(%.4lf) total_cost(%.4lf)\n",
          inputs[0].dim[0], inputs[0].dim[1], inputs[0].dim[2], inputs[0].dim[3],
          runtime, exe_time);
}

// key is (_conv_w)
FuseConvBatchNormAlphaVarKey::FuseConvBatchNormAlphaVarKey(const Tensor& _conv_w)
{
  int idx = 0;
  _conv_w.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(KEY_LENGTH == idx);
}
