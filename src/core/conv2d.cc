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

TensorHandle Graph::group_conv2d(int groups,
                                 const TensorHandle _input,
                                 int _outputC,
                                 int _kernelH, int _kernelW,
                                 int _strideH, int _strideW,
                                 PaddingMode _padding,
                                 ActiMode _activation)
{
  assert(_input->dim[1] % groups == 0);
  assert(_outputC % groups == 0);
  int dims[4] = {_outputC, _input->dim[1] / groups, _kernelH, _kernelW};
  int total = dims[0] * dims[1] * dims[2] * dims[3];
  // Randomly initialize weights
  DATATYPE* data = (DATATYPE*) malloc(total * sizeof(DATATYPE));
  for (int i = 0; i < total; i++)
    data[i] = (DATATYPE)std::rand() / RAND_MAX;
  TensorHandle weight = new_weight(4, dims, data);
  free(data);
/*
  weight.numDim = 4;
  weight.dim[0] = _outputC;
  weight.dim[1] = _input.dim[1] / groups;
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
  return conv2d(_input, weight, _strideH, _strideW, _padding, _activation);
}


TensorHandle Graph::conv2d(const TensorHandle _input,
                           int _outputC,
                           int _kernelH, int _kernelW,
                           int _strideH, int _strideW,
                           PaddingMode _padding,
                           ActiMode _activation)
{
  const int dims[4] = {_outputC, _input->dim[1], _kernelH, _kernelW};
  int total = dims[0] * dims[1] * dims[2] * dims[3];
  // Randomly initialize weights
  DATATYPE* data = (DATATYPE*) malloc(total * sizeof(DATATYPE));
  for (int i = 0; i < total; i++)
    data[i] = (DATATYPE)std::rand() / RAND_MAX;
  TensorHandle weight = new_weight(4, dims, data);
  free(data);
/*
  weight.numDim = 4;
  weight.dim[0] = _outputC;
  weight.dim[1] = _input.dim[1];
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
  return conv2d(_input, weight, _strideH, _strideW,
                _padding, _activation);
}

/*
Tensor Graph::conv2d(Tensor _input, Tensor _weight,
                     int _strideH, int _strideW,
                     PaddingMode _padding,
                     ActiMode _activation)
{
  Op op = model->get_or_create_conv2d(_input, _weight, _strideH, _strideW,
                                      _padding, _activation);
  add_edge(_input.op, op, _input.idx, 0);
  add_edge(_weight.op, op, _weight.idx, 1);
  Tensor t = op.ptr->outputs[0];
  t.op = op;
  return t;
}
*/

TensorHandle Graph::conv2d(const TensorHandle _input,
                           const TensorHandle _weight,
                           int _strideH, int _strideW,
                           PaddingMode _padding,
                           ActiMode _activation)
{
  Op op = model->get_or_create_conv2d(*_input, *_weight, _strideH, _strideW,
                                      _padding, _activation);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  add_edge(_weight->op, op, _weight->idx, 1);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_conv2d(Tensor _input, Tensor _weight,
                               int _strideH, int _strideW,
                               PaddingMode _padding,
                               ActiMode _activation)
{
  if (_input.dim[1] % _weight.dim[1] != 0)
    return Op::INVALID_OP;
  // key is (inputN, inputC, inputH, inputW, outputC, kernelH, kernelW,
  //         strideH, strideW, padding, activation)
  Conv2DKey key(_input, _weight, _strideH, _strideW, _padding, _activation);
  Conv2D* convOp;
  if (conv2d.find(key) != conv2d.end()) {
    convOp = conv2d[key];
  } else {
    convOp = new Conv2D(this, _input, _weight, _strideH, _strideW,
                        _padding, _activation);
    measure_conv2d_cost(convOp);
    conv2d[key] = convOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = convOp;
  return ret;
}

Conv2D::Conv2D(Model* _model, Tensor _input, Tensor _weight,
               int _strideH, int _strideW,
               PaddingMode _padding,
               ActiMode _activation)
: OpBase(_input, _weight, _model, OP_CONV2D),
  strideH(_strideH), strideW(_strideW),
  padding(_padding), activation(_activation)
{
  assert(_input.numDim == 4);
  assert(_weight.numDim == 4);
  //assert(_input.dim[1] == _weight.dim[1]);
  assert(_input.dim[1] % _weight.dim[1] == 0);
  int groups = _input.dim[1] / _weight.dim[1];
  assert(_weight.dim[0] % groups == 0);
  //printf("k(%d %d) pad(%d %d) stride(%d %d)\n",
  //       kernelH, kernelW, padH, padW, strideH, strideW);
  int inputH = _input.dim[2];
  int inputW = _input.dim[3];
  int kernelH = _weight.dim[2];
  int kernelW = _weight.dim[3];
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
  // Set dims and strides
  numOutputs = 1;
  outputs[0].numDim = 4;
  outputs[0].dim[0] = _input.dim[0];
  outputs[0].dim[1] = _weight.dim[0];
  outputs[0].dim[2] = outputH;
  outputs[0].dim[3] = outputW;
  outputs[0].stride[3] = 1;
  outputs[0].stride[2] = outputs[0].stride[3] * outputs[0].dim[3];
  outputs[0].stride[1] = outputs[0].stride[2] * outputs[0].dim[2];
  outputs[0].stride[0] = outputs[0].stride[1] * outputs[0].dim[1];
  // Set SplitInfo
  outputs[0].split[0] = _input.split[0];
  outputs[0].split[1] = _weight.split[0];
  outputs[0].split[2] = _input.split[2];
  outputs[0].split[3] = _input.split[3];
  // Assume we cannot split the H and W dimension,
  // otherwise we need to extend Conv2DKey to include their SplitInfo
  assert(outputs[0].split[2] == SplitInfo::NO_SPLIT);
  assert(outputs[0].split[3] == SplitInfo::NO_SPLIT);
  outputs[0].idx = 0;
}

Conv2D::~Conv2D(void)
{}

bool Conv2D::get_parameter(PMParameter para, int* value)
{
  switch (para) {
    case PM_GROUP:
    {
      int inputC = inputs[0].dim[1];
      int weightC = inputs[1].dim[1];
      assert(inputC % weightC == 0);
      *value = inputC / weightC;
      return true;
    }
    case PM_KERNEL_H:
      *value = inputs[1].dim[2];
      return true;
    case PM_KERNEL_W:
      *value = inputs[1].dim[3];
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
      *value = (int) activation;
      return true;
    default:
      return OpBase::get_parameter(para, value);
  }
}

void Conv2D::get_padding(int* padH, int* padW) {
  int inputH = inputs[0].dim[2];
  int inputW = inputs[0].dim[3];
  int kernelH = inputs[1].dim[2];
  int kernelW = inputs[1].dim[3];
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

void Conv2D::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  size_t outputSize = outputs[0].volume() * sizeof(DATATYPE);
  size_t inputSize = inputs[0].volume() * sizeof(DATATYPE);
  size_t weightSize = inputs[1].volume() * sizeof(DATATYPE);
  // cost metrics
  exe_time += runtime;
  int kernelH = inputs[1].dim[2];
  int kernelW = inputs[1].dim[3];
  int inputC = inputs[1].dim[1];
  flops += outputSize * (kernelH * kernelW * inputC + 1);
  if (activation != AC_MODE_NONE)
    flops += outputSize;
  mem_acc += inputSize + outputSize + weightSize;
  num_kernels += 1;
  printf("        cost[Conv2D]: i(%d %d %d %d) w(%d %d %d %d) s(%d %d) p(%d) cost(%.4lf) total_cost(%.4lf)\n",
          inputs[0].dim[0], inputs[0].dim[1], inputs[0].dim[2], inputs[0].dim[3],
          inputs[1].dim[0], inputs[1].dim[1], inputs[1].dim[2], inputs[1].dim[3],
          strideH, strideW, padding, runtime, exe_time);
}

// keys are (inputN, inputC, inputH, inputW, outputC, kernelH, kernelW,
//           strideH, strideW, padding, acitvation,
//           input.split[0], weight.split[0])
Conv2DKey::Conv2DKey(Tensor _input, Tensor _weight,
                     int _strideH, int _strideW,
                     PaddingMode _padding,
                     ActiMode _activation)
{
  assert(_input.dim[1] % _weight.dim[1] == 0);
  int groups = _input.dim[1] / _weight.dim[1];
  assert(_weight.dim[0] % groups == 0);
  int idx = 0;
  keys[idx++] = _strideH;
  keys[idx++] = _strideW;
  keys[idx++] = _padding;
  keys[idx++] = _activation;
  _input.serialize(keys, idx);
  _weight.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(KEY_LENGTH == idx);
}

