/* Copyright 2018 Stanford
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

#include <unordered_map>
#include "xflow/ops.h"
#include "rules.pb.h"
typedef int TYPE;
#define MAX_SIZE 512
#define MAX_NUM_OPS 8
#define MAX_NUM_TENSORS 8
#define BATCHSIZE 2
#define NO_SAME_INPUTS

using namespace XFlow;
const SplitInfo SplitInfo::NO_SPLIT = SplitInfo();

TYPE relu_function(TYPE input)
{
  return input * (input + 1) + 1;
}

struct TensorTemp {
  int numDim, dim[MAX_DIM], stride[MAX_DIM];
  SplitInfo split[MAX_DIM];
  TYPE data[MAX_SIZE];
  // Do not compare the following metadata for equation checks
  int opIdx, tsIdx;
  inline bool operator==(const TensorTemp& tt) const {
    if (tt.numDim != numDim) return false;
    int total = 1;
    for (int i = 0; i < numDim; i++) {
      if (dim[i] != tt.dim[i]) return false;
      if (stride[i] != tt.stride[i]) return false;
      if (split[i] != tt.split[i]) return false;
      total *= dim[i];
    }
    for (int i = 0; i < total; i++)
      if (data[i] != tt.data[i]) return false;
    return true;
  }
  TensorTemp& operator=(const TensorTemp& tt)
  {
    numDim = tt.numDim;
    int total = 1;
    for (int i = 0; i < numDim; i++) {
      dim[i] = tt.dim[i];
      total *= dim[i];
      stride[i] = tt.stride[i];
      split[i] = tt.split[i];
    }
    assert(total <= MAX_SIZE);
    for (int i = 0; i < total; i++)
      data[i] = tt.data[i];
    opIdx = tt.opIdx;
    tsIdx = tt.tsIdx;
    return *this;
  }
  int size(void)
  {
    int total = 1;
    for (int i = 0; i < numDim; i++)
      total *= dim[i];
    return total;
  }
  inline TYPE get_value(int n, int c, int h, int w) const
  {
    assert(numDim == 4);
    int offset = n * stride[0] + c * stride[1] + h * stride[2] + w * stride[3];
    assert(offset >= 0 && offset < MAX_SIZE);
    return data[offset];
  }
  inline TYPE get_value(int n, int c) const
  {
    assert(numDim == 2);
    int offset = n * stride[0] + c * stride[1];
    assert(offset >= 0 && offset < MAX_SIZE);
    return data[offset];
  }
  inline void set_value(int n, int c, int h, int w, TYPE val)
  {
    assert(numDim == 4);
    int offset = n * stride[0] + c * stride[1] + h * stride[2] + w * stride[3];
    if (offset >= MAX_SIZE) {
      printf("dim = {%d %d %d %d}\n", dim[0], dim[1], dim[2], dim[3]);
      printf("n = %d c = %d h = %d w = %d\n", n, c, h, w);
    }
    assert(offset >= 0 && offset < MAX_SIZE);
    data[offset] = val;
  }
  inline void set_value(int n, int c, TYPE val)
  {
    assert(numDim == 2);
    int offset = n * stride[0] + c * stride[1];
    if (offset >= MAX_SIZE) {
      printf("dim = {%d %d}\n", dim[0], dim[1]);
      printf("n = %d c = %d\n", n, c);
    }
    assert(offset >= 0 && offset < MAX_SIZE);
    data[offset] = val;
  }
  void print(std::string name)
  {
    printf("%s:\n", name.c_str());
    printf("dim[%d] = {%d, %d, %d, %d}\n", numDim, dim[0], dim[1], dim[2], dim[3]);
    printf("stride[%d] = {%d, %d, %d, %d}\n", numDim, stride[0], stride[1], stride[2], stride[3]);
    for (int i = 0; i < size(); i++)
      printf("%d ", data[i]);
    printf("\n");
  }
};

struct TensorTempList {
  int numTensor;
  TensorTemp tensors[MAX_NUM_TENSORS];
  bool operator==(const TensorTempList& ttl) const
  {
    if (numTensor != ttl.numTensor) return false;
    for (int i = 0; i < numTensor; i++)
      if (!(tensors[i] == ttl.tensors[i])) return false;
    return true;
  }
};

class OpTemp {
public:
  OpTemp(int _inputs, int _outputs, OpType _type)
  : numInputs(_inputs), numOutputs(_outputs), type(_type) {}
  virtual bool compute(int n, TensorTemp* inputs, int opIdx) = 0;
  virtual bool compute(const TensorTemp& x, int opIdx) = 0;
  virtual bool compute(const TensorTemp& x, const TensorTemp& y, int opIdx) = 0;
public:
  OpType type;
  int numInputs, numOutputs;
  TensorTemp outputs[MAX_NUM_OUTPUTS];
};

std::map<int, std::string> variable_names;
std::map<const OpTemp*, std::string> operator_names;

struct GraphTemp {
  struct GraphOp {
    const OpTemp* opTemp;
    int opIdx[MAX_NUM_INPUTS], tsIdx[MAX_NUM_INPUTS];
    bool operator==(const GraphOp& gop) const
    {
      if (opTemp != gop.opTemp) return false;
      for (int i = 0; i < opTemp->numInputs; i++) {
        if ((opIdx[i] != gop.opIdx[i]) || (tsIdx[i] != gop.tsIdx[i]))
          return false;
      }
      return true;
    }
  };
  int numOps;
  GraphOp op[MAX_NUM_OPS];
  int mapped_outputs(int* opIdxs, int* tsIdxs) const
  {
    int mappedOutputs = 0;
    for (int op1 = 0; op1 < numOps; op1++)
      for (int ts1 = 0; ts1 < op[op1].opTemp->numOutputs; ts1++) {
        bool found = false;
        for (int op2 = op1 + 1; op2 < numOps; op2++)
          for (int ts2 = 0; ts2 < op[op2].opTemp->numInputs; ts2++)
            if (op[op2].opIdx[ts2] == op1 && op[op2].tsIdx[ts2] == ts1)
              found = true;
        if (!found) {
          opIdxs[mappedOutputs] = op1;
          tsIdxs[mappedOutputs] = ts1;
          mappedOutputs ++;
        }
      }
    return mappedOutputs;
  }
  void push_op(const OpTemp* opTemp)
  {
    assert(opTemp->numInputs == 0);
    op[numOps].opTemp = opTemp;
    numOps ++;
  }
  void push_op(const OpTemp* opTemp, const TensorTemp& tt0)
  {
    assert(opTemp->numInputs == 1);
    op[numOps].opTemp = opTemp;
    op[numOps].opIdx[0] = tt0.opIdx; op[numOps].tsIdx[0] = tt0.tsIdx;
    numOps ++;
  }
  void push_op(const OpTemp* opTemp, const TensorTemp& tt0, const TensorTemp& tt1)
  {
    assert(opTemp->numInputs == 2);
    op[numOps].opTemp = opTemp;
    op[numOps].opIdx[0] = tt0.opIdx; op[numOps].tsIdx[0] = tt0.tsIdx;
    op[numOps].opIdx[1] = tt1.opIdx; op[numOps].tsIdx[1] = tt1.tsIdx;
    numOps ++;
  }
  void pop_op(void)
  {
    numOps --;
  }
  std::string to_string(void)
  {
    //for (int i = 0; i < numOps; i++)
      //printf("[%d] op(%d) input1(%d %d) input2(%d %d)\n", i, op[i].opTemp->type, op[i].opIdx[0], op[i].tsIdx[0], op[i].opIdx[1], op[i].tsIdx[1]);
    std::string name;
    for (int i = numOps - 1; i >= 0; i--)
      for (int j = op[i].opTemp->numOutputs - 1; j >= 0; j--) {
        bool found = false;
        for (int k = i + 1; k < numOps; k++)
          for (int l = 0; l < op[k].opTemp->numInputs; l++)
            if (op[k].opIdx[l] == i && op[k].tsIdx[l] == j)
              found = true;
        if (!found) {
          name = name + to_string(i, j) + " | ";
        }
      }
    return name;
  }
  std::string to_string(int opIdx, int tsIdx)
  {
    if (opIdx < 0) {
      assert(tsIdx == 0);
      assert(variable_names.find(opIdx) != variable_names.end());
      return variable_names[opIdx];
    } else {
      const OpTemp* opTemp = op[opIdx].opTemp;
      assert(operator_names.find(opTemp) != operator_names.end());
      std::string name = operator_names[opTemp] + "["
                       + std::to_string(tsIdx) + "]{";
      for (int i = 0; i < opTemp->numInputs; i++) {
        name = name + "input" + std::to_string(i) + "("
             + to_string(op[opIdx].opIdx[i], op[opIdx].tsIdx[i]) + ")";
      }
      return name + "}";
    }
  }
  int find(std::string name) const
  {
    int idx = 0;
    for (int i = 0; i < numOps; i++) {
      const OpTemp* opTemp = op[i].opTemp;
      for (int j = 0; j < opTemp->numInputs; j++) {
        if (op[i].opIdx[j] < 0) {
          assert(variable_names.find(op[i].opIdx[j]) != variable_names.end());
          if (variable_names[op[i].opIdx[j]] == name)
            return idx;
        }
        idx ++;
      }
    }
    return idx;
  }
  void print(std::string prefix)
  {
    printf("%s\n", prefix.c_str());
    for (int i = 0; i < numOps; i++) {
      const OpTemp* opTemp = op[i].opTemp;
      printf("[%d]  ", opTemp->type);
      for (int j = 0; j < opTemp->numInputs; j++)
        printf("(%d %d) ", op[i].opIdx[j], op[i].tsIdx[j]);
      printf("\n");
    }
  }
};

class ScalarMulTemp : public OpTemp {
public:
  ScalarMulTemp(void)
  : OpTemp(2, 1, OP_MUL)
  {}
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& input, const TensorTemp& scalar, int opIdx)
  {
    if (scalar.numDim != 0) return false;
    outputs[0].numDim = input.numDim;
    int total = 1;
    for (int i = 0; i < input.numDim; i++) {
      outputs[0].dim[i] = input.dim[i];
      outputs[0].stride[i] = input.stride[i];
      outputs[0].split[i] = input.split[i];
      total *= input.dim[i];
    }
    for (int i = 0; i < total; i++)
      outputs[0].data[i] = input.data[i] * scalar.data[0];
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;   
  }
};

class EnlargeConvTemp: public OpTemp {
public:
  EnlargeConvTemp(int _kernelH, int _kernelW)
  : OpTemp(1, 1, OP_ENLARGE), kernelH(_kernelH), kernelW(_kernelW)
  {}
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& input, int opIdx)
  {
    //if (input.opIdx >= 0) return false;
    if (input.numDim != 4) return false;
    if ((input.dim[2] > kernelH) || (input.dim[3] > kernelW)) return false;
    if ((input.dim[2] == kernelH) && (input.dim[3] == kernelW)) return false;
    if (input.opIdx >= 0) return false;
    int offH = (kernelH - input.dim[2]) / 2;
    int offW = (kernelW - input.dim[3]) / 2;
    outputs[0].numDim = 4;
    outputs[0].dim[0] = input.dim[0];
    outputs[0].dim[1] = input.dim[1];
    outputs[0].dim[2] = kernelH;
    outputs[0].dim[3] = kernelW;
    outputs[0].stride[3] = 1;
    outputs[0].stride[2] = outputs[0].stride[3] * outputs[0].dim[3];
    outputs[0].stride[1] = outputs[0].stride[2] * outputs[0].dim[2];
    outputs[0].stride[0] = outputs[0].stride[1] * outputs[0].dim[1];
    outputs[0].split[0] = input.split[0];
    outputs[0].split[1] = input.split[1];
    outputs[0].split[2] = SplitInfo::NO_SPLIT;
    outputs[0].split[3] = SplitInfo::NO_SPLIT;
    if (outputs[0].size() > MAX_SIZE) return false;
    for (int cout = 0; cout < outputs[0].dim[0]; cout++)
      for (int cin = 0; cin < outputs[0].dim[1]; cin++)
        for (int h = 0; h < outputs[0].dim[2]; h++)
          for (int w = 0; w < outputs[0].dim[3]; w++)
            if (h >= offH && w >= offW && h - offH < input.dim[2]
            && w - offW < input.dim[3]) {
              int weightVal = input.get_value(cout, cin, h - offH, w - offW);
              outputs[0].set_value(cout, cin, h, w, weightVal);
            } else {
              outputs[0].set_value(cout, cin, h, w, 0);
            }
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;           
  }
  bool compute(const TensorTemp& input1, const TensorTemp& input2, int opIdx)
  {
    assert(false);
    return false;
  }
public:
  int kernelH, kernelW;
};

class ConstantTemp : public OpTemp {
public:
  ConstantTemp(int _ndim, const int* _dims, OpType _type)
  : OpTemp(0, 1, _type), ndim(_ndim)
  {
    for (int i = 0; i < ndim; i++)
      dims[i] = _dims[i];
  }
  virtual bool compute(int opIdx) = 0;
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& input, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& input1, const TensorTemp& input2, int opIdx)
  {
    assert(false);
    return false;
  }
public:
  int ndim, dims[MAX_DIM];
};

class ConstantPoolTemp : public ConstantTemp {
public:
  ConstantPoolTemp(int _ndim, const int* dims)
  : ConstantTemp(_ndim, dims, OP_CONSTANT_POOL)
  {
    assert(_ndim == 4);
  }
  bool compute(int opIdx)
  {
    outputs[0].numDim = ndim;
    for (int i = ndim-1; i >= 0; i--) {
      outputs[0].dim[i] = dims[i];
      if (i == ndim-1)
        outputs[0].stride[i] = 1;
      else
        outputs[0].stride[i] = outputs[0].stride[i+1] * outputs[0].dim[i+1];
      outputs[0].split[i] = SplitInfo::NO_SPLIT;
    }
    if (outputs[0].size() > MAX_SIZE) return false;
    for (int i = 0; i < outputs[0].size(); i++)
      outputs[0].data[i] = 1; 
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
};

class ConstantIConvTemp : public ConstantTemp {
public:
  ConstantIConvTemp(int _ndim, const int* dims)
  : ConstantTemp(_ndim, dims, OP_CONSTANT_ICONV)
  {
    assert(_ndim == 4);
    assert(dims[0] == dims[1]);
  }
  bool compute(int opIdx)
  {
    outputs[0].numDim = ndim;
    for (int i = ndim-1; i >= 0; i--) {
      outputs[0].dim[i] = dims[i];
      if (i == ndim-1)
        outputs[0].stride[i] = 1;
      else
        outputs[0].stride[i] = outputs[0].stride[i+1] * outputs[0].dim[i+1];
      outputs[0].split[i] = SplitInfo::NO_SPLIT;
    }
    if (outputs[0].size() > MAX_SIZE) return false;
    for (int cout = 0; cout < outputs[0].dim[0]; cout++)
      for (int cin = 0; cin < outputs[0].dim[1]; cin++)
        for (int kh = 0; kh < outputs[0].dim[2]; kh++)
          for (int kw = 0; kw < outputs[0].dim[3]; kw++) {
            if (cout == cin && kh == outputs[0].dim[2]/2 && kw == outputs[0].dim[3]/2)
              outputs[0].set_value(cout, cin, kh, kw, 1);
            else
              outputs[0].set_value(cout, cin, kh, kw, 0);
          }
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
};

class ConstantIMMTemp : public ConstantTemp {
public:
  ConstantIMMTemp(int _ndim, const int* dims)
  : ConstantTemp(_ndim, dims, OP_CONSTANT_IMM)
  {
    assert(_ndim == 2);
    assert(dims[0] == dims[1]);
  }
  bool compute(int opIdx)
  {
    outputs[0].numDim = ndim;
    for (int i = ndim-1; i >= 0; i--) {
      outputs[0].dim[i] = dims[i];
      if (i == ndim-1)
        outputs[0].stride[i] = 1;
      else
        outputs[0].stride[i] = outputs[0].stride[i+1] * outputs[0].dim[i+1];
      outputs[0].split[i] = SplitInfo::NO_SPLIT;
    }
    if (outputs[0].size() > MAX_SIZE) return false;
    for (int cout = 0; cout < outputs[0].dim[0]; cout++)
      for (int cin = 0; cin < outputs[0].dim[1]; cin++)
        outputs[0].data[cout * outputs[0].dim[1] + cin] = cout == cin ? 1 : 0;
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
};

class ConstantOneTemp : public ConstantTemp {
public:
  ConstantOneTemp(int _ndim, const int* dims)
  : ConstantTemp(_ndim, dims, OP_CONSTANT_ONE)
  {
    assert(_ndim == 4);
  }
  bool compute(int opIdx)
  {
    outputs[0].numDim = ndim;
    for (int i = ndim-1; i >= 0; i--) {
      outputs[0].dim[i] = dims[i];
      if (i == ndim-1)
        outputs[0].stride[i] = 1;
      else
        outputs[0].stride[i] = outputs[0].stride[i+1] * outputs[0].dim[i+1];
      outputs[0].split[i] = SplitInfo::NO_SPLIT;
    }
    if (outputs[0].size() > MAX_SIZE) return false;
    for (int i = 0; i < outputs[0].size(); i++)
      outputs[0].data[i] = 1; 
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
};

class Conv2DTemp : public OpTemp {
public:
  Conv2DTemp(int _kernelH, int _kernelW,
             int _strideH, int _strideW,
             bool _samePad, bool _relu)
  : OpTemp(2, 1, OP_CONV2D), kernelH(_kernelH), kernelW(_kernelW),
    strideH(_strideH), strideW(_strideW), samePad(_samePad), relu(_relu)
  {}
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& input, const TensorTemp& weight, int opIdx)
  {
    if (input.numDim != 4 || weight.numDim != 4) return false;
    if ((weight.dim[2] != kernelH) || (weight.dim[3] != kernelW)) return false;
    //if (input.dim[0] != BATCHSIZE && input.dim[0] != 2 * BATCHSIZE) return false;
    //if (input.dim[1] != weight.dim[1]) return false;
    if (input.dim[1] % weight.dim[1] != 0) return false;
    int group = input.dim[1] / weight.dim[1];
    if (weight.dim[0] % group != 0) return false;
    //if (weight.dim[0] == BATCHSIZE) return false;
    int padT, padL;
    if (samePad) {
      outputs[0].numDim = 4;
      outputs[0].dim[0] = input.dim[0];
      outputs[0].dim[1] = weight.dim[0];
      outputs[0].dim[2] = (input.dim[2] + strideH - 1) / strideH;
      outputs[0].dim[3] = (input.dim[3] + strideW - 1) / strideW;
      int padH = max((outputs[0].dim[2] - 1) * strideH + weight.dim[2]
                     - input.dim[2], 0);
      int padW = max((outputs[0].dim[3] - 1) * strideW + weight.dim[3]
                     - input.dim[3], 0);
      padT = padH / 2;
      padL = padW / 2;
    } else {
      outputs[0].numDim = 4;
      outputs[0].dim[0] = input.dim[0];
      outputs[0].dim[1] = weight.dim[0];
      outputs[0].dim[2] = (input.dim[2] - weight.dim[2]) / strideH + 1;
      outputs[0].dim[3] = (input.dim[3] - weight.dim[3]) / strideW + 1;
      padT = 0;
      padL = 0;
    }
    outputs[0].stride[3] = 1;
    outputs[0].stride[2] = outputs[0].stride[3] * outputs[0].dim[3];
    outputs[0].stride[1] = outputs[0].stride[2] * outputs[0].dim[2];
    outputs[0].stride[0] = outputs[0].stride[1] * outputs[0].dim[1];
    outputs[0].split[0] = input.split[0];
    outputs[0].split[1] = weight.split[0];
    outputs[0].split[2] = input.split[2];
    outputs[0].split[3] = input.split[3];

    if (outputs[0].size() > MAX_SIZE) return false;
    for (int n = 0; n < outputs[0].dim[0]; n++)
      for (int c = 0; c < outputs[0].dim[1]; c++)
        for (int h = 0; h < outputs[0].dim[2]; h++)
          for (int w = 0; w < outputs[0].dim[3]; w++) {
            int group_idx = c / (weight.dim[0] / group);
            TYPE val = 0;
            for (int cin = 0; cin < weight.dim[1]; cin ++)
              for (int kh = 0; kh < weight.dim[2]; kh ++)
                for (int kw = 0; kw < weight.dim[3]; kw ++) {
                  int posH = h * strideH + kh - padT;
                  int posW = w * strideW + kw - padL;
                  assert(posH >= -padT && posH <= input.dim[2] + padT);
                  assert(posW >= -padL && posW <= input.dim[3] + padL);
                  if ((posH >= 0) && (posH < input.dim[2])
                  && (posW >= 0) && (posW < input.dim[3])) {
                    int weightVal = weight.get_value(c, cin, kh, kw);
                    int inputVal = input.get_value(n, cin + group_idx * weight.dim[1], posH, posW);
                    val += weightVal * inputVal;
                  }
                }
            if (relu) val = relu_function(val);
            outputs[0].set_value(n, c, h, w, val);
          }
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
public:
  int kernelH, kernelW, strideH, strideW;
  bool relu, samePad;
};

class Pool2DTemp : public OpTemp {
public:
  Pool2DTemp(int _kernelH, int _kernelW,
             int _strideH, int _strideW,
             bool _samePad, OpType _type)
  : OpTemp(1, 1, _type), kernelH(_kernelH), kernelW(_kernelW),
    strideH(_strideH), strideW(_strideW), samePad(_samePad)
  {}
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& input, int opIdx)
  {
    if (input.numDim != 4) return false;
    int padT, padL;
    if (samePad) {
      outputs[0].numDim = 4;
      outputs[0].dim[0] = input.dim[0];
      outputs[0].dim[1] = input.dim[1];
      outputs[0].dim[2] = (input.dim[2] + strideH - 1) / strideH;
      outputs[0].dim[3] = (input.dim[3] + strideW - 1) / strideW;
      int padH = max((outputs[0].dim[2] - 1) * strideH + kernelH
                     - input.dim[2], 0);
      int padW = max((outputs[0].dim[3] - 1) * strideW + kernelW
                     - input.dim[3], 0);
      padT = padH / 2;
      padL = padW / 2;
    } else {
      outputs[0].numDim = 4;
      outputs[0].dim[0] = input.dim[0];
      outputs[0].dim[1] = input.dim[1];
      outputs[0].dim[2] = (input.dim[2] - kernelH) / strideH + 1;
      outputs[0].dim[3] = (input.dim[3] - kernelW) / strideW + 1;
      padT = 0;
      padL = 0;
    }
    outputs[0].stride[3] = 1;
    outputs[0].stride[2] = outputs[0].stride[3] * outputs[0].dim[3];
    outputs[0].stride[1] = outputs[0].stride[2] * outputs[0].dim[2];
    outputs[0].stride[0] = outputs[0].stride[1] * outputs[0].dim[1];
    outputs[0].split[0] = input.split[0];
    outputs[0].split[1] = SplitInfo::NO_SPLIT;
    outputs[0].split[2] = input.split[2];
    outputs[0].split[3] = input.split[3];
    if (outputs[0].size() > MAX_SIZE) return false;
    for (int n = 0; n < outputs[0].dim[0]; n++)
      for (int c = 0; c < outputs[0].dim[1]; c++)
        for (int h = 0; h < outputs[0].dim[2]; h++)
          for (int w = 0; w < outputs[0].dim[3]; w++) {
            TYPE val = 0;
            for (int kh = 0; kh < kernelH; kh++)
              for (int kw = 0; kw < kernelW; kw++) {
                int posH = h * strideH + kh - padT;
                int posW = w * strideW + kw - padL;
                assert(posH >= -padT && posH <= input.dim[2] + padT);
                assert(posW >= -padL && posW <= input.dim[3] + padL);
                if ((posH >= 0) && (posH < input.dim[2])
                && (posW >= 0) && (posW < input.dim[3])) {
                  int inputVal = input.get_value(n, c, posH, posW);
                  if (type == OP_POOL2D_MAX)
                    val = max(inputVal, val);
                  else if (type == OP_POOL2D_AVG)
                    val += inputVal;
                }
              }
            outputs[0].set_value(n, c, h, w, val);
          }
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
  bool compute(const TensorTemp& input, const TensorTemp& weight, int opIdx)
  {
    assert(false);
    return false;
    if (input.numDim != 4 || weight.numDim != 4) return false;
    if ((weight.dim[2] != kernelH) || (weight.dim[3] != kernelW) || (weight.dim[1] != 1)) return false;
    if (input.dim[1] != weight.dim[0]) return false;
    int padT, padL;
    if (samePad) {
      outputs[0].numDim = 4;
      outputs[0].dim[0] = input.dim[0];
      outputs[0].dim[1] = input.dim[1];
      outputs[0].dim[2] = (input.dim[2] + strideH - 1) / strideH;
      outputs[0].dim[3] = (input.dim[3] + strideW - 1) / strideW;
      int padH = max((outputs[0].dim[2] - 1) * strideH + weight.dim[2]
                     - input.dim[2], 0);
      int padW = max((outputs[0].dim[3] - 1) * strideW + weight.dim[3]
                     - input.dim[3], 0);
      padT = padH / 2;
      padL = padW / 2;
    } else {
      outputs[0].numDim = 4;
      outputs[0].dim[0] = input.dim[0];
      outputs[0].dim[1] = input.dim[1];
      outputs[0].dim[2] = (input.dim[2] - weight.dim[2]) / strideH + 1;
      outputs[0].dim[3] = (input.dim[3] - weight.dim[3]) / strideW + 1;
      padT = 0;
      padL = 0;
    }
    outputs[0].stride[3] = 1;
    outputs[0].stride[2] = outputs[0].stride[3] * outputs[0].dim[3];
    outputs[0].stride[1] = outputs[0].stride[2] * outputs[0].dim[2];
    outputs[0].stride[0] = outputs[0].stride[1] * outputs[0].dim[1];
    outputs[0].split[0] = input.split[0];
    outputs[0].split[1] = SplitInfo::NO_SPLIT;
    outputs[0].split[2] = input.split[2];
    outputs[0].split[3] = input.split[3];
    if (outputs[0].size() > MAX_SIZE) return false;
    for (int n = 0; n < outputs[0].dim[0]; n++)
      for (int c = 0; c < outputs[0].dim[1]; c++)
        for (int h = 0; h < outputs[0].dim[2]; h++)
          for (int w = 0; w < outputs[0].dim[3]; w++) {
            TYPE val = 0;
            for (int kh = 0; kh < weight.dim[2]; kh++)
              for (int kw = 0; kw < weight.dim[3]; kw++) {
                int posH = h * strideH + kh - padT;
                int posW = w * strideW + kw - padL;
                assert(posH >= -padT && posH <= input.dim[2] + padT);
                assert(posW >= -padL && posW <= input.dim[3] + padL);
                if ((posH >= 0) && (posH < input.dim[2])
                && (posW >= 0) && (posW < input.dim[3])) {
                  int inputVal = input.get_value(n, c, posH, posW);
                  if (type == OP_POOL2D_MAX)
                    val = max(inputVal, val);
                  else if (type == OP_POOL2D_AVG)
                    val += inputVal;
                }
              }
            outputs[0].set_value(n, c, h, w, val);
          }
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
public:
  int kernelH, kernelW, strideH, strideW;
  bool samePad;
};

class MatmulTemp : public OpTemp {
public:
  MatmulTemp(ActiMode _mode)
  : OpTemp(2, 1, OP_MATMUL), mode(_mode)
  {}
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& input, const TensorTemp& weight, int opIdx)
  {
#ifdef NO_SAME_INPUTS
    if (input == weight) return false;
#endif
    if (input.numDim != 2 || weight.numDim != 2) return false;
    //if (input.dim[0] != BATCHSIZE) return false;
    if (input.dim[1] != weight.dim[0]) return false;
    if (weight.dim[0] == BATCHSIZE) return false;
    outputs[0].numDim = 2;
    outputs[0].dim[0] = input.dim[0];
    outputs[0].dim[1] = weight.dim[1];
    outputs[0].stride[0] = outputs[0].dim[1];
    outputs[0].stride[1] = 1;
    outputs[0].split[0] = input.split[0];
    outputs[0].split[1] = weight.split[1];
    int outputN = outputs[0].dim[0];
    int outputC = outputs[0].dim[1];
    int inputC = input.dim[1];
    for (int i = 0; i < outputN; i++)
      for (int j = 0; j < outputC; j++) {
        TYPE val = 0;
        for (int k = 0; k < inputC; k++)
          val += input.get_value(i, k) * weight.get_value(k, j);
        outputs[0].set_value(i, j, val);
      }
    if (mode == AC_MODE_RELU) {
      for (int i = 0; i < outputN * outputC; i++)
        outputs[0].data[i] = relu_function(outputs[0].data[i]);
    } else if (mode == AC_MODE_SIGMOID) {
      assert(false);
    } else if (mode == AC_MODE_TANH) {
      assert(false);
    } else {
      assert(mode == AC_MODE_NONE);
    }
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
public:
  ActiMode mode;
};

class ElementTemp : public OpTemp {
public:
  ElementTemp(OpType _type)
  : OpTemp(2, 1, _type) {
    assert(_type == OP_EW_ADD || _type == OP_EW_MUL);
  }
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, const TensorTemp& x2, int opIdx)
  {
    if (x1.numDim != x2.numDim) return false;
    int numDim = x1.numDim;
    int total = 1;
    for (int i = 0; i < numDim; i++) {
      if (x1.dim[i] != x2.dim[i])
        return false;
      if (x1.stride[i] != x2.stride[i])
        return false;
      total *= x1.dim[i];
    }
    outputs[0].numDim = numDim;
    for (int i = 0; i < numDim; i++) {
      outputs[0].dim[i] = x1.dim[i];
      outputs[0].stride[i] = x1.stride[i];
      if (x1.split[i] != x2.split[i])
        outputs[0].split[i] = SplitInfo::NO_SPLIT;
      else
        outputs[0].split[i] = x1.split[i];
    }
    if (type == OP_EW_ADD) {
      for (int i = 0; i < total; i++)
        outputs[0].data[i] = x1.data[i] + x2.data[i];
    } else {
      assert(type == OP_EW_MUL);
      for (int i = 0; i < total; i++)
        outputs[0].data[i] = x1.data[i] * x2.data[i];
    }
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
};

class ActivationTemp : public OpTemp {
public:
  ActivationTemp(OpType _type)
  : OpTemp(1, 1, _type) {}
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    outputs[0].numDim = x1.numDim;
    int total = 1;
    for (int i = 0; i < x1.numDim; i++) {
      outputs[0].dim[i] = x1.dim[i];
      outputs[0].stride[i] = x1.stride[i];
      outputs[0].split[i] = x1.split[i];
      total *= x1.dim[i];
    }
    if (type == OP_RELU) {
      for (int i = 0; i < total; i++)
        outputs[0].data[i] = relu_function(x1.data[i]);
    } else if (type == OP_SIGMOID) {
      assert(false);
    } else {
      assert(false);
    }
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
  bool compute(const TensorTemp& x1, const TensorTemp& x2, int opIdx)
  {
    assert(false);
    return false;
  }
};

class TransposeTemp : public OpTemp {
public:
  TransposeTemp(int _n, const int _perm[], bool _shuffle)
  : OpTemp(1, 1, OP_TRANSPOSE), shuffle(_shuffle)
  {
    myNumDim = _n;
    for (int i = 0; i < myNumDim; i++)
      perm[i] = _perm[i];
    for (int i = 0; i < myNumDim; i++) {
      assert(perm[i] >= 0);
      assert(perm[i] < myNumDim);
      for (int j = i + 1; j < myNumDim; j++)
        assert(perm[i] != perm[j]);
    }
  }
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  void dfs(TensorTemp& output, const TensorTemp& x,
           int d, int myPos, int inPos)
  {
    if (d == myNumDim) {
      output.data[myPos] = x.data[inPos];
    } else {
      for (int i = 0; i < output.dim[d]; i++) {
        dfs(output, x, d+1, myPos + i * output.stride[d], inPos + i * x.stride[perm[d]]);
      }
    }
  }
  bool compute(const TensorTemp& x, int opIdx)
  {
    if (x.numDim != myNumDim) return false;
    outputs[0].numDim = myNumDim;
    for (int i = 0; i < myNumDim; i++) {
      outputs[0].dim[i] = x.dim[perm[i]];
      outputs[0].split[i] = x.split[perm[i]];
    }
    if (shuffle) {
      int size = 1;
      for (int i = myNumDim - 1; i >= 0; i--) {
        outputs[0].stride[i] = size;
        size *= outputs[0].dim[i];
      }
    } else {
      for (int i = 0; i < myNumDim; i++)
        outputs[0].stride[i] = x.stride[perm[i]];
    }
    dfs(outputs[0], x, 0, 0, 0);
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
  bool compute(const TensorTemp& x1, const TensorTemp& x2, int opIdx)
  {
    assert(false);
    return false;
  }
  int myNumDim, perm[MAX_DIM];
  int idx;
  bool shuffle;
};

class ConcatTemp : public OpTemp {
public:
  ConcatTemp(int n, int _numDim, int _axis)
  : OpTemp(n, 1, OP_CONCAT), myNumDim(_numDim), axis(_axis)
  {}
  void dfs(TensorTemp& output, const TensorTemp& input,
           int d, int myPos, int inPos) {
    if (d == myNumDim) {
      output.data[myPos] = input.data[inPos];
    } else {
      for (int i = 0; i < input.dim[d]; i++) {
        if (d == axis)
          dfs(output, input, d+1,
              myPos + (i + accAxis) * output.stride[d],
              inPos + i * input.stride[d]);
        else
          dfs(output, input, d+1,
              myPos + i * output.stride[d],
              inPos + i * input.stride[d]);
      }
    }
  }
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    for (int i = 0; i < n; i++)
      if (inputs[i].numDim != myNumDim) return false;
    if (axis >= myNumDim) return false;
    for (int i = 1; i < n; i++)
      for (int j = 0; j < myNumDim; j++)
        if ((j != axis) && (inputs[0].dim[j] != inputs[i].dim[j]))
          return false;
    outputs[0].numDim = myNumDim;
    for (int i = 0; i < myNumDim; i++) {
      outputs[0].dim[i] = inputs[0].dim[i];
      outputs[0].split[i] = inputs[0].split[i];
      if (i != axis) {
        for (int j = 1; j < n; j++)
          outputs[0].split[i].combine(inputs[j].split[i]);
      } else {
        for (int j = 1; j < n; j++) {
          outputs[0].split[i].merge(outputs[0].dim[i], inputs[j].split[i]);
          outputs[0].dim[i] += inputs[j].dim[i];
        }
      }
    }
    if (outputs[0].size() > MAX_SIZE)
      return false;
    int size = 1;
    for (int i = myNumDim - 1; i >= 0; i--) {
      outputs[0].stride[i] = size;
      size = size * outputs[0].dim[i];
    }
    accAxis = 0;
    for (int i = 0; i < n; i++) {
      dfs(outputs[0], inputs[i], 0, 0, 0);
      accAxis += inputs[0].dim[axis];
    }
    /*
    int outSize = 1, inSize = 1;
    for (int i = 0; i < axis; i++)
      outSize *= inputs[0].dim[i];
    for (int i = axis + 1; i < myNumDim; i++)
      inSize *= inputs[0].dim[i];
    int outIdx = 0, inIdxs[MAX_NUM_INPUTS];
    for (int i = 0; i < n; i++)
      inIdxs[i] = 0;
    for (int out = 0; out < outSize; out++)
      for (int i = 0; i < n; i++)
        for (int j = 0; j < inputs[i].dim[axis]; j++)
          for (int in = 0; in < inSize; in++) {
            outputs[0].data[outIdx++] = inputs[i].data[inIdxs[i]++];
          }
    assert(outIdx == outputs[0].size());
    */
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, const TensorTemp& x2, int opIdx)
  {
    assert(numInputs == 2);
    TensorTemp xs[2];
    xs[0] = x1;
    xs[1] = x2;
    return compute(2, xs, opIdx);
  }
public:
  int myNumDim, axis;
  int accAxis;
};

class SplitTemp : public OpTemp {
public:
  SplitTemp(int n, int _axis)
  : OpTemp(1, n, OP_SPLIT), axis(_axis)
  {}
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  void dfs(TensorTemp& output, const TensorTemp& input,
           int d, int myPos, int inPos)
  {
    if (d == output.numDim) {
      output.data[myPos] = input.data[inPos];
    } else {
      for (int i = 0; i < output.dim[d]; i++) {
        if (d == axis) {
          dfs(output, input, d + 1,
              myPos + i * output.stride[d],
              inPos + (i + accAxis) * input.stride[d]);
        } else {
          dfs(output, input, d + 1,
              myPos + i * output.stride[d],
              inPos + i * input.stride[d]);
        }
      }
    }
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    // TODO:Only consider 2-split for now
    assert(numOutputs == 2);
    if (x1.split[axis].num == 0) return false;
    SplitInfo parent = x1.split[axis], left, right;
    int oldPos = x1.dim[axis], curPos;
    for (int i = numOutputs - 1; i >= 0; i--) {
      outputs[i].numDim = x1.numDim;
      int size = 1;
      for (int j = x1.numDim - 1; j >= 0; j--) {
        if (j != axis) {
          outputs[i].dim[j] = x1.dim[j];
          outputs[i].split[j] = x1.split[j];
        } else {
          if (i > 0)
            parent.divide(left, right, curPos);
          else {
            curPos = 0;
            right = parent;
          }
          outputs[i].dim[j] = oldPos - curPos;
          oldPos = curPos;
          parent = left;
          outputs[i].split[j] = right;
        }
        outputs[i].stride[j] = size;
        size = size * outputs[i].dim[j];
      }
      accAxis = oldPos;
      dfs(outputs[i], x1, 0, 0, 0);
      /*
      int outSize = 1;
      int inSize = 1;
      for (int j = 0; j < axis; j++)
        outSize = outSize * outputs[i].dim[j];
      for (int j = axis; j < outputs[i].numDim; j++)
        inSize = inSize * outputs[i].dim[j];
      for (int out = 0; out < outSize; out++) {
        int dstIdx = out * inSize, srcIdx = out * inSize * numOutputs + inSize * i;
        for (int in = 0; in < inSize; in++)
          outputs[i].data[dstIdx++] = x1.data[srcIdx++];
      }
      */
      outputs[i].opIdx = opIdx;
      outputs[i].tsIdx = i;
    }
    return true;
  }
  bool compute(const TensorTemp& x1, const TensorTemp& x2, int opIdx)
  {
    assert(false);
    return false;
  }
public:
  int axis, accAxis;
};

namespace std {
  template <>
  struct hash<SplitInfo>
  {
    size_t operator()(const SplitInfo& si) const
    {
      size_t res = 17;
      res = res * 31 + hash<int>()(si.num);
      for (int i = 0; i < si.num; i++)
        res = res * 31 + hash<int>()(si.pos[i]);
      return res;
    }
  };
  template <>
  struct hash<TensorTemp>
  {
    size_t operator()(const TensorTemp& tt) const
    {
      size_t res = 17;
      int total = 1;
      res = res * 31 + hash<int>()(tt.numDim);
      for (int i = 0; i < tt.numDim; i++) {
        res = res * 31 + hash<int>()(tt.dim[i]);
        res = res * 31 + hash<int>()(tt.stride[i]);
        res = res * 31 + hash<SplitInfo>()(tt.split[i]);
        total *= tt.dim[i];
      }
      for (int i = 0; i < total; i++)
        res = res * 31 + hash<TYPE>()(tt.data[i]);
      return res;
    }
  };

  template <>
  struct hash<TensorTempList>
  {
    size_t operator()(const TensorTempList& ttl) const
    {
      size_t res = 17;
      res = res * 31 + hash<int>()(ttl.numTensor);
      for (int i = 0; i < ttl.numTensor; i++)
        res = res * 31 + hash<TensorTemp>()(ttl.tensors[i]);
      return res;
    }
  };
}

bool find_same_subgraph(const GraphTemp::GraphOp& o1,
                        const GraphTemp::GraphOp& o2)
{
  if (o1.opTemp != o2.opTemp)
    return false;
  for (int i = 0; i < o1.opTemp->numInputs; i++) {
    if ((o1.opIdx[i] != o2.opIdx[i]) || (o1.tsIdx[i] != o2.tsIdx[i])) return false;
    if (o1.opIdx[i] >= 0) return false;
  }
  return true;
}

bool find_same_supergraph(const GraphTemp::GraphOp& o1,
                          const GraphTemp::GraphOp& o2)
{
  if (o1.opTemp != o2.opTemp)
    return false;
  // Only one input is different
  int diff = 0;
  for (int i = 0; i < o1.opTemp->numInputs; i++) {
    if ((o1.opIdx[i] != o2.opIdx[i]) || (o1.opIdx[i] >= 0))
      diff ++;
  }
  if (diff > 1) return false;
  return true;
}

bool variable_ordering(const GraphTemp& g)
{
  if (g.find("x1") > g.find("x2")) return false;
  if (g.find("x2") > g.find("x3")) return false;
  if (g.find("w1") > g.find("w2")) return false;
  if (g.find("w2") > g.find("w3")) return false;
  if (g.find("i1") > g.find("i2")) return false;
  if (g.find("i2") > g.find("i3")) return false;
  if (g.find("w4") > g.find("w5")) return false;
  if (g.find("w5") > g.find("w6")) return false;
  return true;
}

bool pass_checks(const GraphTemp& g1,
                 const GraphTemp& g2)
{
  // Pruning: cannot have common subgraphs
  for (int i = 0; i < g1.numOps; i++)
    for (int j = 0; j < g2.numOps; j++)
      if (find_same_subgraph(g1.op[i], g2.op[j]))
        return false;
  // Pruning: cannot have common supergraphs
  if (find_same_supergraph(g1.op[g1.numOps-1], g2.op[g2.numOps-1]))
    return false;
  // Pruning: check variable ordering (x1 used before x2 before x3)
  if ((!variable_ordering(g1)) && (!variable_ordering(g2)))
    return false;
  // Pruning: variable renaming (e.g., "x1" must appear before "x2")
  return true;
}

bool same_via_subst(const GraphTemp& g1,
                    const GraphTemp& g2,
                    std::map<int, int>& variable_subst)
{
  if (g1.numOps != g2.numOps) return false;
  for (int i = 0; i < g1.numOps; i++) {
    if (g1.op[i].opTemp != g2.op[i].opTemp) return false;
    for (int j = 0; j < g1.op[i].opTemp->numInputs; j++) {
      if (g1.op[i].tsIdx[j] != g2.op[i].tsIdx[j]) return false;
      int op1 = g1.op[i].opIdx[j];
      int op2 = g2.op[i].opIdx[j];
      if ((op1 >= 0) || (op2 >= 0)) {
        if (op1 != op2) return false;
      } else {
        if (variable_subst.find(op1) == variable_subst.end()) {
          variable_subst[op1] = op2;
        } else {
          if (variable_subst[op1] != op2) return false;
        }
      }
    }
  }
  return true;
}

struct TransferTemp {
  GraphTemp fstGraph, sndGraph;
  bool isDuplicate;
};

size_t graph_guid = 0;
void dfs(int depth,
         GraphTemp& graph,
         std::vector<TensorTemp>& inputs,
         const std::vector<OpTemp*>& ops,
         std::unordered_map<size_t, GraphTemp>& hashmap,
         std::vector<TransferTemp>& transfers)
{
  // Pruning should not have duplicated tensors
  for (int i = 0; i < inputs.size(); i++)
    for (int j = i + 1; j < inputs.size(); j++) {
      if (inputs[i] == inputs[j])
        return;
    }
  // Pruning should not have duplicated operators
  for (int i = 0; i < graph.numOps; i++)
    for (int j = i + 1; j < graph.numOps; j++) {
      if (graph.op[i] == graph.op[j])
        return;
    }
  // Add current subgraph to graphs
  TensorTempList ttl;
  ttl.numTensor = 0;
  for (int i = inputs.size() - 1; inputs[i].opIdx >= 0; i--) {
    bool found = false;
    for (int j = 0; j < graph.numOps; j++)
      for (int k = 0; k < graph.op[j].opTemp->numInputs; k++)
        if (graph.op[j].opIdx[k] == inputs[i].opIdx
        && graph.op[j].tsIdx[k] == inputs[i].tsIdx)
          found = true;
    if (!found) {
      ttl.numTensor++;
      assert(ttl.numTensor <= MAX_NUM_TENSORS);
      ttl.tensors[ttl.numTensor-1] = inputs[i];
    }
  }
  graph_guid ++;
  if (graph_guid % 100000 == 0)
    printf("Num of Graphs = %zu hashmap.size() = %zu\n", graph_guid, hashmap.size()); 
  size_t hashKey = hash<TensorTempList>()(ttl);
  if (hashmap.find(hashKey) != hashmap.end()) {
    // Found a match
    GraphTemp oldgraph = hashmap[hashKey];
    if (pass_checks(oldgraph, graph)) {
      // Pruning: cannot have redundant transfers via variable substitutions
      bool found = false;
      for (int i = 0; i < transfers.size(); i++)
        if (!(transfers[i].isDuplicate)) {
          // first->oldgraph, second->graph
          {
            std::map<int, int> variable_subst;
            if (same_via_subst(transfers[i].fstGraph, oldgraph, variable_subst)
            && same_via_subst(transfers[i].sndGraph, graph, variable_subst)) {
              found = true;
              break;
            }
          }
          // first-> graph, second-> oldgraph
          {
            std::map<int, int> variable_subst;
            if (same_via_subst(transfers[i].fstGraph, graph, variable_subst)
            && same_via_subst(transfers[i].sndGraph, oldgraph, variable_subst)) {
              found = true;
              break;
            }
          }
          // oldgraph->first, graph->second
          {
            std::map<int, int> variable_subst;
            if (same_via_subst(oldgraph, transfers[i].fstGraph, variable_subst)
            && same_via_subst(graph, transfers[i].sndGraph, variable_subst)) {
              transfers[i].isDuplicate = true;
              continue;
            }
          }
          // graph->first, oldgraph->second
          {
            std::map<int, int> variable_subst;
            if (same_via_subst(graph, transfers[i].fstGraph, variable_subst)
            && same_via_subst(oldgraph, transfers[i].sndGraph, variable_subst)) {
              transfers[i].isDuplicate = true;
              continue;
            }
          }
        } // if (it->second)
      if (!found) {
        TransferTemp tt;
        tt.fstGraph = oldgraph;
        tt.sndGraph = graph;
        tt.isDuplicate = false;
        transfers.push_back(tt);
        printf("Source Graph: %s\n", oldgraph.to_string().c_str());
        printf("Target Graph: %s\n", graph.to_string().c_str());
      }
    }
  } else {
    hashmap[hashKey] = graph;
  }
  if (depth >= 3) return; // MAX_NUM_OPS
  for (int i = 0; i < ops.size(); i++)
    switch (ops[i]->type) {
      case OP_EW_ADD:
      case OP_EW_MUL:
      {
        OpTemp* op = ops[i];
        for (int j = 0; j < inputs.size(); j++)
          for (int k = j + 1; k < inputs.size(); k++)
            if (op->compute(inputs[j], inputs[k], depth)) {
              inputs.push_back(op->outputs[0]);
              graph.push_op(op, inputs[j], inputs[k]);
              dfs(depth + 1, graph, inputs, ops, hashmap, transfers);
              graph.pop_op();
              inputs.pop_back();
            }
        break;
      }
      case OP_CONV2D:
      {
        OpTemp* op = ops[i];
        for (int j = 0; j < inputs.size(); j++) {
          if (inputs[j].opIdx < 0 && variable_names[inputs[j].opIdx][0] == 'w')
            continue;
          for (int k = 0; k < inputs.size(); k++)
            if (op->compute(inputs[j], inputs[k], depth)) {
              inputs.push_back(op->outputs[0]);
              graph.push_op(op, inputs[j], inputs[k]);
              dfs(depth + 1, graph, inputs, ops, hashmap, transfers);
              graph.pop_op();
              inputs.pop_back();
            }
        }
        break;
      }
      case OP_MATMUL:
      {
        OpTemp* op = ops[i];
        for (int j = 0; j < inputs.size(); j++)
          for (int k = 0; k < inputs.size(); k++)
            if (op->compute(inputs[j], inputs[k], depth)) {
              inputs.push_back(op->outputs[0]);
              graph.push_op(op, inputs[j], inputs[k]);
              dfs(depth + 1, graph, inputs, ops, hashmap, transfers);
              graph.pop_op();
              inputs.pop_back();
            }
        break;
      }
      case OP_MUL:
      {
        OpTemp* op = ops[i];
        for (int j = 0; j < inputs.size(); j++)
          for (int k = 0; k < inputs.size(); k++)
            if ((j != k) && (op->compute(inputs[j], inputs[k], depth))) {
              inputs.push_back(op->outputs[0]);
              graph.push_op(op, inputs[j], inputs[k]);
              dfs(depth + 1, graph, inputs, ops, hashmap, transfers);
              graph.pop_op();
              inputs.pop_back();
            }
        break;
      }
      case OP_RELU:
      case OP_ENLARGE:
      case OP_TRANSPOSE:
      case OP_POOL2D_AVG:
      case OP_POOL2D_MAX:
      {
        OpTemp* op = ops[i];
        for (int j = 0; j < inputs.size(); j++)
          if (op->compute(inputs[j], depth)) {
            inputs.push_back(op->outputs[0]);
            graph.push_op(op, inputs[j]);
            dfs(depth + 1, graph, inputs, ops, hashmap, transfers);
            graph.pop_op();
            inputs.pop_back();
          }
        break;
      }
      case OP_CONSTANT_IMM:
      case OP_CONSTANT_ICONV:
      case OP_CONSTANT_ONE:
      case OP_CONSTANT_POOL:
      {
        ConstantTemp* op = (ConstantTemp*) ops[i];
        if (op->compute(depth)) {
          inputs.push_back(op->outputs[0]);
          graph.push_op(op);
          dfs(depth + 1, graph, inputs, ops, hashmap, transfers);
          graph.pop_op();
          inputs.pop_back();
        }
        break;
      }
      case OP_CONCAT:
      {
        OpTemp* op = ops[i];
        assert(op->numInputs == 2);
        for (int j = 0; j < inputs.size(); j++)
          for (int k = j + 1; k < inputs.size(); k++)
            if (op->compute(inputs[j], inputs[k], depth)) {
              inputs.push_back(op->outputs[0]);
              graph.push_op(op, inputs[j], inputs[k]);
              dfs(depth + 1, graph, inputs, ops, hashmap, transfers);
              graph.pop_op();
              inputs.pop_back();
            }
        break;
      }
      case OP_SPLIT:
      {
        OpTemp* op = ops[i];
        for (int j = 0; j < inputs.size(); j++)
          if (op->compute(inputs[j], depth)) {
            for (int k = 0; k < op->numOutputs; k++)
              inputs.push_back(op->outputs[k]);
            graph.push_op(op, inputs[j]);
            dfs(depth + 1, graph, inputs, ops, hashmap, transfers);
            graph.pop_op();
            for (int k = 0; k < op->numOutputs; k++)
              inputs.pop_back();
          }
        break;
      }
      default:
        assert(false);
    }
}

void init_tensor_temp(TensorTemp& tt, std::string name, int opIdx, int tsIdx, int n = 0, int c = 0, int h = 0, int w = 0)
{
  variable_names[opIdx] = name;
  tt.numDim = 0;
  if (n > 0) { tt.numDim ++; tt.dim[0] = n; tt.split[0] = SplitInfo::NO_SPLIT;}
  if (c > 0) { tt.numDim ++; tt.dim[1] = c; tt.split[1] = SplitInfo::NO_SPLIT;}
  if (h > 0) { tt.numDim ++; tt.dim[2] = h; tt.split[2] = SplitInfo::NO_SPLIT;}
  if (w > 0) { tt.numDim ++; tt.dim[3] = w; tt.split[3] = SplitInfo::NO_SPLIT;}
  int size = 1;
  for (int i = tt.numDim - 1; i >= 0; i --) {
    tt.stride[i] = size;
    size *= tt.dim[i];
  }
  tt.opIdx = opIdx;
  tt.tsIdx = tsIdx;
  int total = tt.size();
  assert(total <= MAX_SIZE);
  for (int i = 0; i < total; i++)
    //tt.data[i] = (std::rand() % 2000 - 1000);
    tt.data[i] = std::rand() - RAND_MAX / 2;
}

void init_graph_temp(GraphTemp& graph)
{
  graph.numOps = 0;
}

void pb_fill_parameter(int key, int value,
                       GraphSubst::Operator* pbOp)
{
  GraphSubst::Parameter* para = pbOp->add_para();
  para->set_key(key);
  para->set_value(value);
}

void pb_fill_op(const GraphTemp::GraphOp& graphOp,
                GraphSubst::Operator* pbOp)
{
  pbOp->set_type(graphOp.opTemp->type);
  for (int j = 0; j < graphOp.opTemp->numInputs; j++) {
    GraphSubst::Tensor* tensor = pbOp->add_input();
    tensor->set_opid(graphOp.opIdx[j]);
    tensor->set_tsid(graphOp.tsIdx[j]);   
  }
  switch (graphOp.opTemp->type) {
    case OP_CONV2D:
    {
      Conv2DTemp* conv = (Conv2DTemp*) graphOp.opTemp;
      PaddingMode padding = conv->samePad ?
          PD_MODE_SAME : PD_MODE_VALID;
      ActiMode activation = conv->relu ?
          ActiMode(AC_MODE_RELU) : ActiMode(AC_MODE_NONE);
      pb_fill_parameter(PM_KERNEL_H, conv->kernelH, pbOp);
      pb_fill_parameter(PM_KERNEL_W, conv->kernelW, pbOp);
      pb_fill_parameter(PM_STRIDE_H, conv->strideH, pbOp);
      pb_fill_parameter(PM_STRIDE_W, conv->strideW, pbOp);
      pb_fill_parameter(PM_PAD, padding, pbOp);
      pb_fill_parameter(PM_ACTI, activation, pbOp);
      break;
    }
    case OP_POOL2D_AVG:
    case OP_POOL2D_MAX:
    {
      Pool2DTemp* pool = (Pool2DTemp*) graphOp.opTemp;
      PaddingMode padding = pool->samePad ?
          PD_MODE_SAME: PD_MODE_VALID;
      pb_fill_parameter(PM_OP_TYPE, pool->type, pbOp);
      pb_fill_parameter(PM_KERNEL_H, pool->kernelH, pbOp);
      pb_fill_parameter(PM_KERNEL_W, pool->kernelW, pbOp);
      pb_fill_parameter(PM_STRIDE_H, pool->strideH, pbOp);
      pb_fill_parameter(PM_STRIDE_W, pool->strideW, pbOp);
      pb_fill_parameter(PM_PAD, padding, pbOp);
      pb_fill_parameter(PM_ACTI, AC_MODE_NONE, pbOp);
      break;
    }
    case OP_CONCAT:
    {
      ConcatTemp* concat = (ConcatTemp*) graphOp.opTemp;
      pb_fill_parameter(PM_NUM_INPUTS, concat->numInputs, pbOp);
      pb_fill_parameter(PM_AXIS, concat->axis, pbOp);
      pb_fill_parameter(PM_NUMDIM, concat->myNumDim, pbOp);
      break;
    }
    case OP_SPLIT:
    {
      SplitTemp* split = (SplitTemp*) graphOp.opTemp;
      pb_fill_parameter(PM_NUM_OUTPUTS, split->numOutputs, pbOp);
      pb_fill_parameter(PM_AXIS, split->axis, pbOp);
      break;
    }
    case OP_RELU:
    case OP_SIGMOID:
    case OP_MUL:
    case OP_EW_ADD:
    case OP_EW_MUL:
    {
      break;
    }
    case OP_ENLARGE:
    {
      EnlargeConvTemp* enlarge = (EnlargeConvTemp*) graphOp.opTemp;
      pb_fill_parameter(PM_KERNEL_H, enlarge->kernelH, pbOp);
      pb_fill_parameter(PM_KERNEL_W, enlarge->kernelW, pbOp);
      break;
    }
    case OP_CONSTANT_ICONV:
    case OP_CONSTANT_POOL:
    {
      ConstantTemp* constant = (ConstantTemp*) graphOp.opTemp;
      assert(constant->ndim == 4);
      pb_fill_parameter(PM_KERNEL_H, constant->dims[2], pbOp);
      pb_fill_parameter(PM_KERNEL_W, constant->dims[3], pbOp);
      break;
    }
    case OP_CONSTANT_ONE:
    case OP_CONSTANT_IMM:
    {
      ConstantTemp* constant = (ConstantTemp*) graphOp.opTemp;
      break;
    }
    case OP_MATMUL:
    {
      MatmulTemp* matmul = (MatmulTemp*) graphOp.opTemp;
      pb_fill_parameter(PM_ACTI, matmul->mode, pbOp);
      break;
    }
    case OP_TRANSPOSE:
    {
      TransposeTemp* transpose = (TransposeTemp*) graphOp.opTemp;
      int ndim = transpose->myNumDim;
      pb_fill_parameter(PM_NUMDIM, ndim, pbOp);
      int perm = 0;
      for (int i = 0; i < ndim; i++)
        perm = perm * ndim + transpose->perm[i];
      pb_fill_parameter(PM_PERM, perm, pbOp);
      pb_fill_parameter(PM_OUTSHUFFLE, transpose->shuffle, pbOp);
      break;
    }
    case OP_BATCHNORM:
    default:
      printf("unsupported type(%d)\n", graphOp.opTemp->type);
      assert(false);
  }
}

void pb_fill_rule(const GraphTemp& fstGraph,
                  const GraphTemp& sndGraph,
                  GraphSubst::Rule* rule)
{
  int srcOpIdxs[MAX_NUM_TENSORS], srcTsIdxs[MAX_NUM_TENSORS];
  int dstOpIdxs[MAX_NUM_TENSORS], dstTsIdxs[MAX_NUM_TENSORS];
  int srcMappedOutputs = fstGraph.mapped_outputs(srcOpIdxs, srcTsIdxs);
  int dstMappedOutputs = sndGraph.mapped_outputs(dstOpIdxs, dstTsIdxs);
  assert(srcMappedOutputs == dstMappedOutputs);
  for (int i = 0; i < fstGraph.numOps; i++) {
    GraphSubst::Operator* srcOp = rule->add_srcop();
    pb_fill_op(fstGraph.op[i], srcOp);
  }
  for (int i = 0; i < sndGraph.numOps; i++) {
    GraphSubst::Operator* dstOp = rule->add_dstop();
    pb_fill_op(sndGraph.op[i], dstOp);
  }
  for (int i = 0; i < srcMappedOutputs; i++) {
    GraphSubst::MapOutput* mapOutput = rule->add_mappedoutput();
    mapOutput->set_srcopid(srcOpIdxs[i]);
    mapOutput->set_dstopid(dstOpIdxs[i]);
    mapOutput->set_srctsid(srcTsIdxs[i]);
    mapOutput->set_dsttsid(dstTsIdxs[i]);
  }
}

int main(int argc, char **argv)
{
  std::unordered_map<size_t, GraphTemp> hashmap;
  std::vector<TransferTemp> transfers;
  std::vector<TensorTemp> inputs;
  GraphTemp graph;
  init_graph_temp(graph);
  // Create 2D tensors
  TensorTemp x1, x2, x3, w1, w2, w3;
  init_tensor_temp(x1, "x1", -1, 0, BATCHSIZE, 4);
  inputs.push_back(x1);
  init_tensor_temp(x2, "x2", -2, 0, BATCHSIZE, 4);
  inputs.push_back(x2);
  init_tensor_temp(x3, "x3", -3, 0, BATCHSIZE, 4);
  inputs.push_back(x3);
  init_tensor_temp(w1, "w1", -4, 0, 4, 4);
  inputs.push_back(w1);
  init_tensor_temp(w2, "w2", -5, 0, 4, 4);
  inputs.push_back(w2);
  init_tensor_temp(w3, "w3", -6, 0, 4, 4);
  inputs.push_back(w3);
  // Create 4D tensors
  TensorTemp i1, i2, i3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14;
  init_tensor_temp(i1, "i1", -7, 0, BATCHSIZE, 4, 5, 5);
  inputs.push_back(i1);
  init_tensor_temp(i2, "i2", -8, 0, BATCHSIZE, 4, 5, 5);
  inputs.push_back(i2);
  init_tensor_temp(i3, "i3", -9, 0, BATCHSIZE, 4, 5, 5);
  inputs.push_back(i3);
  init_tensor_temp(w4, "w4", -10, 0, 4, 4, 3, 3);
  inputs.push_back(w4);
  init_tensor_temp(w5, "w5", -11, 0, 4, 4, 3, 3);
  inputs.push_back(w5);
  init_tensor_temp(w6, "w6", -12, 0, 4, 4, 3, 3);
  inputs.push_back(w6);
  init_tensor_temp(w7, "w7", -13, 0, 4, 4, 1, 3);
  inputs.push_back(w7);
  init_tensor_temp(w8, "w8", -14, 0, 4, 4, 1, 3);
  inputs.push_back(w8);
  init_tensor_temp(w9, "w9", -15, 0, 4, 4, 1, 3);
  inputs.push_back(w9);
  init_tensor_temp(w10, "w10", -16, 0, 4, 4, 3, 1);
  inputs.push_back(w10);
  init_tensor_temp(w11, "w12", -17, 0, 4, 4, 3, 1);
  inputs.push_back(w11);
  init_tensor_temp(w12, "w12", -18, 0, 4, 4, 3, 1);
  inputs.push_back(w12);
  init_tensor_temp(w13, "w13", -19, 0, 4, 1, 3, 3);
  inputs.push_back(w13);

  // Create 0D scalar tensors
  TensorTemp s0;
  init_tensor_temp(s0, "s0", -20, 0);
  inputs.push_back(s0);
  std::vector<OpTemp*> ops;
  ops.push_back(new MatmulTemp(AC_MODE_NONE));
  operator_names[ops.back()] = "MatMul";
  ops.push_back(new ElementTemp(OP_EW_ADD));
  operator_names[ops.back()] = "EWAdd";
  ops.push_back(new ElementTemp(OP_EW_MUL));
  operator_names[ops.back()] = "EWMul";
  ops.push_back(new Conv2DTemp(3, 3, 1, 1, true, false));
  operator_names[ops.back()] = "Conv3x3S";
  ops.push_back(new Conv2DTemp(3, 3, 1, 1, true, true));
  operator_names[ops.back()] = "Conv3x3SR";
  ops.push_back(new Conv2DTemp(1, 1, 1, 1, true, false));
  operator_names[ops.back()] = "Conv1x1S";
  ops.push_back(new Conv2DTemp(1, 1, 1, 1, true, true));
  operator_names[ops.back()] = "Conv1x1SR";
  ops.push_back(new Conv2DTemp(1, 3, 1, 1, true, false));
  operator_names[ops.back()] = "Conv1x3S";
  ops.push_back(new Conv2DTemp(1, 3, 1, 1, true, true));
  operator_names[ops.back()] = "Conv1x3SR";
  //ops.push_back(new Conv2DTemp(3, 1, 1, 1, true, false));
  //operator_names[ops.back()] = "Conv3x1S";
  //ops.push_back(new Conv2DTemp(3, 1, 1, 1, true, true));
  //operator_names[ops.back()] = "Conv3x1SR";
  ops.push_back(new Pool2DTemp(3, 3, 1, 1, true, OP_POOL2D_AVG));
  operator_names[ops.back()] = "Pool3x3SA";
  ops.push_back(new Pool2DTemp(3, 3, 1, 1, true, OP_POOL2D_MAX));
  operator_names[ops.back()] = "Pool3x3SM";

  ops.push_back(new ConstantPoolTemp(w13.numDim, w13.dim));
  operator_names[ops.back()] = "Constant_Pool";
  ops.push_back(new ConstantIConvTemp(w4.numDim, w4.dim));
  operator_names[ops.back()] = "Constant_IConv";
  ops.push_back(new ConstantIMMTemp(w1.numDim, w1.dim));
  operator_names[ops.back()] = "Constant_IMM";
  ops.push_back(new ConstantOneTemp(i1.numDim, i1.dim));
  operator_names[ops.back()] = "Constant_One";
  ops.push_back(new EnlargeConvTemp(3, 3));
  operator_names[ops.back()] = "Enlarge3x3";
  ops.push_back(new ScalarMulTemp());
  operator_names[ops.back()] = "ScalarMul";
  ops.push_back(new ActivationTemp(OP_RELU));
  operator_names[ops.back()] = "Relu";
  ops.push_back(new ConcatTemp(2/*n*/, 2/*numDim*/, 1/*axis*/));
  operator_names[ops.back()] = "Concat_1";
  ops.push_back(new ConcatTemp(2/*n*/, 2/*numDim*/, 0/*axis*/));
  operator_names[ops.back()] = "Concat_0";
  ops.push_back(new ConcatTemp(2/*n*/, 4/*numDim*/, 1/*axis*/));
  operator_names[ops.back()] = "Concat_1";
  ops.push_back(new ConcatTemp(2/*n*/, 4/*numDim*/, 0/*axis*/));
  operator_names[ops.back()] = "Concat_0";
  ops.push_back(new SplitTemp(2/*n*/, 1/*axis*/));
  operator_names[ops.back()] = "Split_1";
  ops.push_back(new SplitTemp(2/*n*/, 0/*axis*/));
  operator_names[ops.back()] = "Split_0";
  const int trans10[2] = {1, 0};
  // Should enable shuffle = true one
  ops.push_back(new TransposeTemp(2/*n*/, trans10, false/*shuffle*/));
  operator_names[ops.back()] = "Transpose_10";
  ops.push_back(new TransposeTemp(2/*n*/, trans10, true/*shuffle*/));
  operator_names[ops.back()] = "TransposeShuffle_10";
#ifdef DEADCODE
  // <test1>
  MatmulTemp* matmul = new MatmulTemp(AC_MODE_NONE);
  TransposeTemp* transpose = new TransposeTemp(2, trans10, false);
  TransposeTemp* transposeT = new TransposeTemp(2, trans10, true);
  TensorTemp o1, o2, o3, o4, o5;
  assert(matmul->compute(w1, w2, 0));
  o1 = matmul->outputs[0];
  assert(transpose->compute(w1, 0));
  o2 = transpose->outputs[0];
  assert(transpose->compute(w2, 1));
  o3 = transpose->outputs[0];
  assert(matmul->compute(o3, o2, 2));
  o4 = matmul->outputs[0];
  assert(transposeT->compute(o4, 3));
  o5 = transposeT->outputs[0];
  o1.print("o1");
  w1.print("w1");
  o2.print("o2");
  w2.print("w2");
  o3.print("o3");
  o4.print("o4");
  o5.print("o5");
  assert(o1 == o5);
#endif
#ifdef DEADCODE
  // <test2>
  MatmulTemp* matmul = new MatmulTemp(AC_MODE_NONE);
  SplitTemp* split = new SplitTemp(2/*n*/, 1/*axis*/);
  ConcatTemp* concat = new ConcatTemp(2/*n*/, 2/*numDim*/, 1/*axis*/);
  TensorTemp o1, o2, o3, o4, o5, o6;
  assert(matmul->compute(x1, w1, 0));
  o1 = matmul->outputs[0];
  assert(matmul->compute(x1, w2, 1));
  o2 = matmul->outputs[0];
  assert(concat->compute(w1, w2, 0));
  o3 = concat->outputs[0];
  assert(matmul->compute(x1, o3, 1));
  o4 = matmul->outputs[0];
  assert(split->compute(o4, 2));
  o5 = split->outputs[0];
  o6 = split->outputs[1];
  o1.print("o1");
  o4.print("o4");
  o5.print("o5");
  assert(o1 == o5);
  assert(o2 == o6);
#endif
  // <test3>
  ConstantPoolTemp* constant = new ConstantPoolTemp(w13.numDim, w13.dim);
  Pool2DTemp* pool = new Pool2DTemp(3, 3, 1, 1, true, OP_POOL2D_AVG);
  Conv2DTemp* conv = new Conv2DTemp(3, 3, 1, 1, true, false);
  TensorTemp o1, o2, o3;
  assert(pool->compute(i1, 0));
  o1 = pool->outputs[0];
  assert(constant->compute(0));
  o2 = constant->outputs[0];
  assert(conv->compute(i1, o2, 1));
  o3 = conv->outputs[0];
  o1.print("o1");
  o3.print("o3");
  assert(o1 == o3);
  // <test4>
  ConstantIConvTemp* constant2 = new ConstantIConvTemp(w4.numDim, w4.dim);
  assert(constant2->compute(0));
  o1 = constant2->outputs[0];
  assert(conv->compute(i1, o1, 1));
  o2 = conv->outputs[0];
  o2.print("o2");
  i1.print("i1");
  assert(o2 == i1);
  // <test5>
  MatmulTemp* matmul = new MatmulTemp(AC_MODE_NONE);
  ConstantIMMTemp* constant3 = new ConstantIMMTemp(w1.numDim, w1.dim);
  assert(constant3->compute(0));
  o1 = constant3->outputs[0];
  assert(matmul->compute(x1, o1, 1));
  o2 = matmul->outputs[0];
  assert(o2 == x1);
  // <test6>
  ElementTemp* ew_mul = new ElementTemp(OP_EW_MUL);
  ConstantOneTemp* constant4 = new ConstantOneTemp(i1.numDim, i1.dim);
  assert(constant4->compute(0));
  o1 = constant4->outputs[0];
  assert(ew_mul->compute(i1, o1, 1));
  o2 = ew_mul->outputs[0];
  assert(o2 == i1);

  dfs(0, graph, inputs, ops, hashmap, transfers);
  printf("===================== Generated %zu Transfers =====================\n", transfers.size());
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  GraphSubst::RuleCollection collection;
  size_t count = 0;
  for (int i = 0; i < transfers.size(); i++)
    if (!(transfers[i].isDuplicate)) {
      count ++;
      printf("Source Graph: %s\n", transfers[i].fstGraph.to_string().c_str());
      printf("Target Graph: %s\n", transfers[i].sndGraph.to_string().c_str());
      pb_fill_rule(transfers[i].fstGraph, transfers[i].sndGraph, collection.add_rule());
    }
  std::fstream outputFile("graph_subst.pb", ios::out | ios::trunc);
  collection.SerializeToOstream(&outputFile);
  google::protobuf::ShutdownProtobufLibrary();
  printf("===================== Generated %zu Transfers =====================\n", count);
  return 0;
}
