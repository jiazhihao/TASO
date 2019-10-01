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

TensorHandle Graph::pool2d_max(const TensorHandle _input,
                               int _kernelH, int _kernelW,
                               int _strideH, int _strideW,
                               PaddingMode _padding,
                               ActiMode _activation)
{
  int num = _input->dim[1] * _kernelH * _kernelW;
  DATATYPE* data_ptr = (DATATYPE*) malloc(num * sizeof(DATATYPE));
  for (int i = 0; i < num; i++)
    data_ptr[i] = 1.0f / (_kernelH * _kernelW);
  const int dims[4] = {_input->dim[1], 1, _kernelH, _kernelW};
  TensorHandle weight = new_weight(4, dims, data_ptr);
/*
  weight.numDim = 4;
  weight.dim[0] = _input.dim[1];
  weight.dim[1] = 1;
  weight.dim[2] = _kernelH;
  weight.dim[3] = _kernelW;
  weight.stride[3] = 1;
  weight.stride[2] = weight.stride[3] * weight.dim[3];
  weight.stride[1] = weight.stride[2] * weight.dim[2];
  weight.stride[0] = weight.stride[1] * weight.dim[1];
  weight.op.guid = GUID_WEIGHT;
  weight.op.ptr = NULL;
  weight.idx = 0;
  weight = noop(weight);
*/
  Op op = model->get_or_create_pool2d(
              *_input, *weight, OP_POOL2D_MAX, _kernelH, _kernelW,
              _strideH, _strideW, _padding, _activation);
  add_edge(_input->op, op, _input->idx, 0);
  add_edge(weight->op, op, weight->idx, 1);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

TensorHandle Graph::pool2d_avg(const TensorHandle _input,
                               int _kernelH, int _kernelW,
                               int _strideH, int _strideW,
                               PaddingMode _padding,
                               ActiMode _activation)
{
  int num = _input->dim[1] * _kernelH * _kernelW;
  DATATYPE* data_ptr = (DATATYPE*) malloc(num * sizeof(DATATYPE));
  for (int i = 0; i < num; i++)
    data_ptr[i] = 1.0f / (_kernelH * _kernelW);
  const int dims[4] = {_input->dim[1], 1, _kernelH, _kernelW};
  TensorHandle weight = new_weight(4, dims, data_ptr);
/*
  weight.numDim = 4;
  weight.dim[0] = _input.dim[1];
  weight.dim[1] = 1;
  weight.dim[2] = _kernelH;
  weight.dim[3] = _kernelW;
  weight.stride[3] = 1;
  weight.stride[2] = weight.stride[3] * weight.dim[3];
  weight.stride[1] = weight.stride[2] * weight.dim[2];
  weight.stride[0] = weight.stride[1] * weight.dim[1];
  weight.op.guid = GUID_WEIGHT;
  weight.op.ptr = NULL;
  weight.idx = 0;
  weight = noop(weight);
*/
  Op op = model->get_or_create_pool2d(
              *_input, *weight, OP_POOL2D_AVG, _kernelH, _kernelW,
              _strideH, _strideW, _padding, _activation);
  add_edge(_input->op, op, _input->idx, 0);
  add_edge(weight->op, op, weight->idx, 1);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_pool2d(Tensor _input, Tensor _weight,
                               OpType _type,
                               int _kernelH, int _kernelW,
                               int _strideH, int _strideW,
                               PaddingMode _padding,
                               ActiMode _activation)

{
  // keys are (inputN, inputC, inputH, inputW, kernelH, kernelW,              
  //           strideH, strideW, padding, activation, _type)
  Pool2DKey key(_input, _type, _kernelH, _kernelW, _strideH, _strideW,
                _padding, _activation);
  Pool2D* poolOp;
  if (pool2d.find(key) != pool2d.end()) {
    poolOp = pool2d[key];
  } else {
    poolOp = new Pool2D(this, _input, _weight, _type, _kernelH, _kernelW,
                        _strideH, _strideW, _padding, _activation);
    measure_pool2d_cost(poolOp);
    pool2d[key] = poolOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = poolOp;
  return ret;
}

Pool2D::Pool2D(Model* _model, Tensor _input,
               Tensor _weight, OpType _type,
               int _kernelH, int _kernelW,
               int _strideH, int _strideW,
               PaddingMode _padding,
               ActiMode _activation)
: OpBase(_input, _weight, _model, _type),
  kernelH(_kernelH), kernelW(_kernelW),
  strideH(_strideH), strideW(_strideW), 
  padding(_padding), activation(_activation)
{
  assert(type == OP_POOL2D_MAX || type == OP_POOL2D_AVG);
  assert(_input.numDim == 4);
  int inputC = _input.dim[1];
  int inputH = _input.dim[2];
  int inputW = _input.dim[3];
  int outputH, outputW;
  switch (padding)
  {
    case PD_MODE_SAME:
      outputH = (inputH + strideH - 1) / strideH;
      outputW = (inputW + strideW - 1) / strideW;
      break;
    case PD_MODE_VALID:
      outputH = (inputH - kernelH) / strideH + 1;
      outputW = (inputW - kernelW) / strideW + 1;
      break;
    default:
      assert(false);
  }
  //int outputH = 1 + (inputH + 2 * padH - kernelH) / strideH;
  //int outputW = 1 + (inputW + 2 * padW - kernelW) / strideW;
  //printf("k(%d %d) padding(%d) s(%d %d) o(%d %d)\n",
  //       kernelH, kernelW, padding, strideH, strideW, outputH, outputW);
  numOutputs = 1;
  outputs[0].numDim = 4;
  outputs[0].dim[0] = _input.dim[0];
  outputs[0].dim[1] = _input.dim[1];
  outputs[0].dim[2] = outputH;
  outputs[0].dim[3] = outputW;
  // Set strides
  outputs[0].stride[3] = 1;
  outputs[0].stride[2] = outputs[0].dim[3] * outputs[0].stride[3];
  outputs[0].stride[1] = outputs[0].dim[2] * outputs[0].stride[2];
  outputs[0].stride[0] = outputs[0].dim[1] * outputs[0].stride[1];
  // Set SplitInfo
  outputs[0].split[0] = _input.split[0];
  outputs[0].split[1] = _input.split[1];
  outputs[0].split[2] = SplitInfo::NO_SPLIT;
  outputs[0].split[3] = SplitInfo::NO_SPLIT;
  outputs[0].idx = 0;
}

Pool2D::~Pool2D(void)
{
}

bool Pool2D::get_parameter(PMParameter para, int* value)
{
  switch (para) {
    case PM_KERNEL_H:
      *value = kernelH;
      return true;
    case PM_KERNEL_W:
      *value = kernelW;
      return true;
    case PM_STRIDE_H:
      *value = strideH;
      return true;
    case PM_STRIDE_W:
      *value = strideW;
      return true;
    case PM_PAD:
      *value = padding;
      return true;
    case PM_ACTI:
      *value = activation;
      return true;
    default:
      return OpBase::get_parameter(para, value);
  }
}

void Pool2D::get_padding(int* padH, int* padW) {
  int inputH = inputs[0].dim[2];
  int inputW = inputs[0].dim[3];
  // TODO eliminate duplicated code with conv2d version
  // Reference: https://www.tensorflow.org/api_guides/python/nn#Convolution
  switch (padding) {
    case PD_MODE_SAME:
      int totalPadH, totalPadW;
      if (inputH % strideH == 0)
        totalPadH = max(kernelH - strideH, 0);
      else
        totalPadH = max(kernelH - (inputH % strideH), 0);
      if (inputW % strideW == 0)
        totalPadW = max(kernelW - strideW, 0);
      else
        totalPadW = max(kernelW - (inputW % strideW), 0);
      // assert same padding on both sides
      *padH = (totalPadH + 1) / 2;
      *padW = (totalPadW + 1) / 2;
      break;
    case PD_MODE_VALID:
      *padH = 0;
      *padW = 0;
      break;
    default:
      assert(false);
  }
}

void Pool2D::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  int outputSize = 1, inputSize = 1;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  for (int i = 0; i < inputs[0].numDim; i++)
    inputSize *= inputs[0].dim[i];
  // cost metrics
  exe_time += runtime;
  flops += outputSize * kernelH * kernelW;
  mem_acc += inputSize;
  num_kernels += 1;
  printf("        cost[Pool2D]: i(%d %d %d %d) k(%d %d) s(%d %d) cost(%.4lf) total_cost(%.4lf)\n",
         inputs[0].dim[0], inputs[0].dim[1], inputs[0].dim[2], inputs[0].dim[3],
         kernelH, kernelW, strideH, strideW, runtime, exe_time);
}

// keys are (kernelH, kernelW, strideH, strideW, padding, activation, _type,
//           input)
Pool2DKey::Pool2DKey(Tensor _input, OpType _type,
                     int _kernelH, int _kernelW, int _strideH, int _strideW,
                     PaddingMode _padding,
                     ActiMode _activation)
{
  int idx = 0;
  keys[idx++] = _kernelH;
  keys[idx++] = _kernelW;
  keys[idx++] = _strideH;
  keys[idx++] = _strideW;
  keys[idx++] = _padding;
  keys[idx++] = _activation;
  keys[idx++] = _type;
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(KEY_LENGTH == idx);
}

