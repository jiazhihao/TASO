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
#ifndef _CNN_OPS_H_
#define _CNN_OPS_H_

#ifdef USE_CUDNN
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifdef TRT
#include "NvInfer.h"
#include "NvUtils.h"

using namespace nvinfer1;
#endif

#ifdef USE_DNNL
#include "dnnl.hpp"
using DNNLNet = std::vector<std::pair<dnnl::primitive, std::unordered_map<int, dnnl::memory>>>;
#endif

#include <cassert>
#include <map>
#include <array>
#include <vector>
#include <set>
#include <list>
#include <iostream>
#include <fstream>
#include <memory>
using namespace std;

namespace taso {

#define MAX_DIM 8
#define MAX_NUM_SPLITS 32
#define MAX_NUM_INPUTS 6
#define MAX_NUM_OUTPUTS 6
#define BATCH_SIZE 1
#define MAX_TENSOR_SIZE 512 * 1024 * 1024 // 512MB
#define REPEAT_TIMES 32
#define WARMUP_TIMES 8
const size_t WORK_SPACE_SIZE = (size_t)2 * 1024 * 1024 * 1024; // 2GB
typedef float DATATYPE;

class Model;
class OpBase;

enum {
  GUID_INVALID = 0,
  GUID_INPUT = 10,
  GUID_WEIGHT = 11,
  GUID_PRESERVED = 19,
};

//This must be consistent with python/taso/_cython/CCore.pxd
enum OpType {
  OP_INPUT,
  OP_WEIGHT,
  OP_ANY,
  OP_CONV2D,
  OP_DROPOUT,
  OP_LINEAR,
  OP_POOL2D_MAX,
  OP_POOL2D_AVG,
  OP_RELU,
  OP_SIGMOID,
  OP_TANH,
  OP_BATCHNORM,
  OP_CONCAT,
  OP_SPLIT,
  OP_RESHAPE,
  OP_TRANSPOSE,
  OP_EW_ADD,
  OP_EW_MUL,
  OP_MATMUL,
  OP_MUL,
  OP_ENLARGE,
  OP_MERGE_GCONV,
  OP_CONSTANT_IMM,
  OP_CONSTANT_ICONV,
  OP_CONSTANT_ONE,
  OP_CONSTANT_POOL,
  OP_SQUEEZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze
  OP_UNSQUEEZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze
  OP_EW_SUB, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub
  OP_EW_DIV, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div
  OP_EW_EQUAL, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal
  OP_EW_GREATER, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Greater
  OP_EW_LESS, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Less
  OP_EW_MAX, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Max
  OP_EW_MIN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min
  OP_REDUCE_ARGMAX, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax
  OP_REDUCE_ARGMIN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMin
  OP_REDUCE_MAX, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMax
  OP_REDUCE_MEAN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMean
  OP_REDUCE_MIN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMin
  OP_REDUCE_PROD, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceProd
  OP_REDUCE_SUM, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum
  OP_PAD, //https://github.com/dmlc/tvm/blob/master/topi/python/topi/nn/pad.py
  OP_SHAPE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape
  OP_SIZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Size
  OP_TOPK, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
  OP_WHERE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Where
  OP_CEIL, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil
  OP_CAST, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast
  OP_EXP, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp
  OP_ROUND, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round
  OP_LOG, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log
  OP_LOGICAL_NOT, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Not
  OP_SQRT, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt
  OP_LEAKYRELU,
  OP_SLICE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice
  OP_RESIZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
  OP_PRELU, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#PRelu
};

struct Op {
  Op(void);
  Op(size_t _guid, OpBase* _ptr)
  : guid(_guid), ptr(_ptr) {}
  inline bool operator==(const Op& b) const {
    if (guid != b.guid) return false;
    if (ptr != b.ptr) return false;
    return true;
  }
  inline bool operator!=(const Op& b) const {
    if (guid != b.guid) return true;
    if (ptr != b.ptr) return true;
    return false;
  }
  inline bool operator<(const Op& b) const {
    if (guid != b.guid) return guid < b.guid;
    if (ptr != b.ptr) return ptr < b.ptr;
    return true;
  }
  Op& operator=(const Op& op)
  {
    guid = op.guid;
    ptr = op.ptr;
    return *this;
  }
  std::string op_to_string(const OpBase* ptr);
  std::string to_string(void)
  {
    if (ptr != NULL) {
      return op_to_string(ptr) + "_" + std::to_string(guid);
    }
    else {
      return "UnmappedOp_" + std::to_string(guid);
    }
  }
  static const Op INVALID_OP;
  size_t guid;
  OpBase* ptr;
};

struct Edge {
  Edge(void);
  Edge(Op _srcOp, Op _dstOp, int _srcIdx, int _dstIdx);
  Op srcOp, dstOp;
  int srcIdx, dstIdx;
};

struct EdgeCompare {
  bool operator()(const Edge& a, const Edge& b) const {
    if (!(a.srcOp == b.srcOp)) return a.srcOp < b.srcOp;
    if (!(a.dstOp == b.dstOp)) return a.dstOp < b.dstOp;
    if (a.srcIdx != b.srcIdx) return a.srcIdx < b.srcIdx;
    if (a.dstIdx != b.dstIdx) return a.dstIdx < b.dstIdx;
    return false;
  };
};

struct SrcEdge {
  SrcEdge(int _idx, Op _op);
  int idx;
  Op op;
};

struct SrcEdgeCompare {
  bool operator()(const SrcEdge& a, const SrcEdge& b) const {
    if (!(a.op == b.op)) return a.op < b.op;
    if (a.idx != b.idx) return a.idx < b.idx;
    return false;
  };
};

struct OpCompare {
  bool operator()(const Op& a, const Op& b) const {
    if (a.guid != b.guid) return a.guid < b.guid;
    return a.ptr < b.ptr;
  };
};

struct SplitInfo {
  SplitInfo(void) {num = 0;}
  inline bool operator==(const SplitInfo& rhs) const {
    if (num != rhs.num) return false;
    for (int i = 0; i < num; i++)
      if (pos[i] != rhs.pos[i])
        return false;
    return true;
  }
  void merge(int offset, const SplitInfo& next) {
    if (num + 1 + next.num >= MAX_NUM_SPLITS) {
      printf("num = %d, next.num = %d\n", num, next.num);
    }
    assert(num + 1 + next.num < MAX_NUM_SPLITS);
    for (int i = 0; i < next.num; i++)
      pos[num++] = offset + next.pos[i];
    pos[num++] = offset;
  }
  inline bool operator!=(const SplitInfo& rhs) const
  {
    if (num != rhs.num) return true;
    for (int i = 0; i < num; i++)
      if (pos[i] != rhs.pos[i]) return true;
    return false;
  }
  SplitInfo& operator=(const SplitInfo& st)
  {
    num = st.num;
    for (int i = 0; i < num; i++)
      pos[i] = st.pos[i];
    return *this;
  }
  void divide(SplitInfo& left, SplitInfo& right, int &mid) {
    assert(num > 0);
    left.num = 0;
    right.num = 0;
    mid = pos[num - 1];
    int idx = 0;
    while (idx < num && pos[idx] < mid)
      left.pos[left.num++] = pos[idx++];
    while (idx < num - 1)
      right.pos[right.num++] = pos[idx++];
  }
  void combine(const SplitInfo& next) {
    if (num != next.num)
      num = 0;
    for (int i = 0; i < num; i++)
      if (pos[i] != next.pos[i]) {
        num = 0;
        return;
      }
  }
  void serialize(int* keys, int& idx) const {
    keys[idx++] = num;
    for (int i = 0; i < num; i++)
      keys[idx++] = pos[i];
  }
  static const SplitInfo NO_SPLIT;
  int num;
  int pos[MAX_NUM_SPLITS];
};

struct Tensor {
  static const int MAX_KEY_LENGTH = (MAX_NUM_SPLITS + 2) * MAX_DIM + 2;
  static const int MAGIC_NUMBER = 23333;
  Tensor(void)
  : numDim(0), idx(0), op(), data_ptr(NULL) {
    for (int i = 0; i < MAX_DIM; i++)
      split[i].num = 0;
  }
  Tensor(int ndim, const int* dims, size_t guid, DATATYPE* data = NULL)
  : numDim(ndim), idx(0), op(guid, NULL), data_ptr(data) {
    assert(guid != GUID_INVALID);
    assert(ndim <= MAX_DIM);
    int count = 1;
    for (int i = ndim-1; i >= 0; i--) {
      dim[i] = dims[i];
      stride[i] = count;
      count *= dim[i];
      split[i]  = SplitInfo::NO_SPLIT;
    }
  }
  Tensor& operator=(const Tensor& src) {
    numDim = src.numDim;
    for (int i = 0; i < numDim; i++) {
      dim[i] = src.dim[i];
      stride[i] = src.stride[i];
      split[i] = src.split[i];
    }
    idx = src.idx;
    op = src.op;
    data_ptr = src.data_ptr;
    return *this;
  }
  int volume(void) const {
    int ret = 1;
    for (int i = 0; i < numDim; i++)
      ret *= dim[i];
    return ret;
  }
  std::string to_string(std::string name)
  {
    name = name + "(";
    for (int i = 0; i < numDim; i++) {
      std::string suffix = (i == numDim -1) ? ")" : " ";
      name = name + std::to_string(dim[i]) + ":"
           + std::to_string(stride[i]) + suffix;
    }
    return name;
  }
  void serialize(int* keys, int& idx) const
  {
    keys[idx++] = MAGIC_NUMBER;
    keys[idx++] = numDim;
    for (int i = 0; i < numDim; i++)
      keys[idx++] = dim[i];
    for (int i = 0; i < numDim; i++)
      keys[idx++] = stride[i];
    for (int i = 0; i < numDim; i++)
      split[i].serialize(keys, idx);
  }
  bool has_same_shape_stride_split(const Tensor& tensor) const
  {
    if (numDim != tensor.numDim)
      return false;
    for (int i = 0; i < numDim; i++) {
      if (dim[i] != tensor.dim[i])
        return false;
      if (stride[i] != tensor.stride[i])
        return false;
      if (split[i] != tensor.split[i])
        return false;
    }
    return true;
  }
  bool default_layout(void) const
  {
    int cnt = 1;
    for (int i = numDim-1; i >= 0; i--) {
      if (stride[i] != cnt) return false;
      cnt *= dim[i];
    }
    return true;
  }
  //bool operator==(const Tensor& b);
  int numDim, dim[MAX_DIM], stride[MAX_DIM];
  int idx; // idx is used for Ops with multiple outputs (e.g., split)
  Op op;
  void* data_ptr;
  // Meta data for splits
  SplitInfo split[MAX_DIM];
};

//typedef shared_ptr<Tensor> TensorHandle;
typedef Tensor* TensorHandle;

enum DataType {
  DT_FLOAT = 111,
  DT_DOUBLE = 222,
  DT_HALF = 333,
  DT_INT8 = 444,
  DT_UINT8 = 555,
  DT_INT32 = 666,
  DT_INT64 = 777,
  DT_BOOL = 888,
};

//This must be consistent with python/taso/_cython/CCore.pxd
enum PMParameter {
  PM_OP_TYPE,   	// AnyOp
  PM_NUM_INPUTS,	// AnyOp
  PM_NUM_OUTPUTS,	// AnyOp
  PM_GROUP,             // Conv2D
  PM_KERNEL_H,		// Conv2D, Pool2D
  PM_KERNEL_W,		// Conv2D, Pool2D
  PM_STRIDE_H,		// Conv2D, Pool2D
  PM_STRIDE_W,		// Conv2D, Pool2D
  PM_PAD,		// Conv2D, Pool2D
  PM_ACTI,		// Conv2D, Pool2D
  PM_NUMDIM,		// Concat, Transpose
  PM_AXIS,		// Concat, Split
  PM_PERM,		// Transpose
  PM_OUTSHUFFLE,	// Transpose
  PM_MERGE_GCONV_COUNT, // MergeGConv
  PM_AXES,		// Squeeze, Unsqueeze, Reduce*
  PM_KEEP_DIMS,         // Reduce*
};

enum TNParameter {
  IN_0 = 100,
  IN_1 = 101,
  IN_2 = 102,
  IN_3 = 103,
  IN_4 = 104,
  IN_5 = 105,
  OU_0 = 200,
  OU_1 = 201,
  OU_2 = 202,
  OU_3 = 203,
  OU_4 = 204,
  OU_5 = 205,
};

enum DIMParameter {
  DIM_0 = 300,
  DIM_1 = 301,
  DIM_2 = 302,
  DIM_3 = 303,
  DIM_ND = 310,
};

//That this must be consistent with python/taso/_cython/CCore.pxd
enum ActiMode {
  AC_MODE_NONE,
  AC_MODE_SIGMOID,
  AC_MODE_RELU,
  AC_MODE_TANH, 
};

//That this must be consistent with python/taso/_cython/CCore.pxd
enum PaddingMode {
  PD_MODE_SAME,
  PD_MODE_VALID,
};

//That this must be consistent with python/taso/_cython/CCore.pxd
//enum ConstantMode {
//  CN_MODE_IDENTITY,
//  CN_MODE_ZEROS,
//  CN_MODE_ONES,
//  CN_MODE_ONES_SCALED_L1,
//  CN_MODE_ONES_SCALED_L2,
//  CN_MODE_ONES_SCALED_ALL,
//};

class OpBase {
public:
  OpBase(Model* _model, OpType _type); // No inputs
  OpBase(const Tensor& input, Model* _model, OpType _type);
  OpBase(const Tensor& input0, const Tensor& input1,
         Model* _model, OpType _type);
  OpBase(const Tensor& input0, const Tensor& input1, const Tensor& input2,
         Model* _model, OpType _type);
  OpBase(const Tensor& input0, const Tensor& input1,
         const Tensor& input2, const Tensor& input3,
         const Tensor& input4, Model* _model, OpType _type);
  OpBase(int n, Tensor* inputs, Model* _model, OpType _type);
  virtual bool get_input_parameter(TNParameter, DIMParameter, int*);
  virtual bool get_int_parameter(PMParameter, int*);
  //virtual bool get_float_parameter(PMParameter, float*);
  //virtual bool get_ints_parameter(PMParameter, std::vector<int>*);
  virtual void forward(bool block = false) = 0;
  virtual void map(void) = 0;
  virtual void unmap(void) = 0;
  virtual void collect_costs(float& exe_time, float& flops,
                             float& mem_acc, int& num_kernels) = 0;
public:
  Tensor inputs[MAX_NUM_INPUTS], outputs[MAX_NUM_OUTPUTS];
  int numInputs, numOutputs;
  Model *model;
  OpType type;
  float runtime;
#ifdef USE_DNNL
  DNNLNet net;
#endif
};

class Graph {
public:
  Graph();
  TensorHandle new_input(int dim, const int* dims);
  TensorHandle new_weight(int dim, const int* dims, const DATATYPE* data);
  TensorHandle new_weight(const Tensor& input);
  void add_edge(Op srcOp, Op dstOp, int srcIdx, int dstIdx);
  void remove_edge(Edge e);
  bool has_edge(Op srcOp, Op dstOp, int srcIdx, int dstIdx);
  void replace_node(Op oldOp, Op newOp);
  void remove_node(Op oldOp);
  void export_to_file(std::string file_name);
  // This conv2ds will create a weight tensor
  TensorHandle group_conv2d(int groups,
                            const TensorHandle _input,
                            int _outputC,
                            int _kernelH, int _kernelW,
                            int _strideH, int strideW,
                            PaddingMode _padding,
                            ActiMode _activation = AC_MODE_NONE);
  TensorHandle batchnorm(const TensorHandle _input,
                         const TensorHandle _scale,
                         const TensorHandle _bias,
                         const TensorHandle _mean,
                         const TensorHandle _var);
  TensorHandle cast(const TensorHandle _input, DataType _datatype);
  TensorHandle ceil(const TensorHandle _input);
  TensorHandle concat(int axis, int n, const TensorHandle* _inputs);
  TensorHandle constant(int ndim, int* dims, OpType _type);
  TensorHandle conv2d(const TensorHandle _input,
                      int _outputC,
                      int _kernelH, int _kernelW,
                      int _strideH, int _strideW,
                      PaddingMode _padding,
                      ActiMode _activation = AC_MODE_NONE);
  TensorHandle conv2d(const TensorHandle _input,
                      const TensorHandle _weight,
                      int _strideH, int _strideW,
                      PaddingMode _padding,
                      ActiMode _activation = AC_MODE_NONE);
  TensorHandle dropout(const TensorHandle _input);
  TensorHandle element(OpType type,
                       const TensorHandle _t1,
                       const TensorHandle _t2);
  TensorHandle elementwise_unary(const TensorHandle _input, OpType _type);
  TensorHandle enlarge(const TensorHandle _w1, const TensorHandle _w2);
  TensorHandle exp(const TensorHandle _input);
  TensorHandle fc(const TensorHandle _input,
                  int _outputC,
                  ActiMode _actiMode = AC_MODE_NONE);
  TensorHandle leakyrelu(const TensorHandle _input, float _alpha,
                         bool _inplace=true);
  TensorHandle log(const TensorHandle _input);
  TensorHandle logical_not(const TensorHandle _input);
  TensorHandle matmul(const TensorHandle _input,
                      const TensorHandle _weight,
                      ActiMode _actiMode = AC_MODE_NONE);
  TensorHandle merge_gconv(const TensorHandle _weight, int count);
  TensorHandle mul(const TensorHandle _x,
                   const TensorHandle _y);
  TensorHandle pad(const TensorHandle _input,
                   const std::vector<int>& _pad_before,
                   const std::vector<int>& _pad_after,
                   float _pad_value);
  TensorHandle pool2d_max(const TensorHandle _input,
                          int _kernelH, int _kernelW,
                          int _strideH, int _strideW,
                          PaddingMode _padding,
                          ActiMode _activation = AC_MODE_NONE);
  TensorHandle pool2d_avg(const TensorHandle _input,
                          int _kernelH, int _kernelW,
                          int _strideH, int _strideW,
                          PaddingMode _padding,
                          ActiMode _activation = AC_MODE_NONE);
  TensorHandle reduce(const TensorHandle _input,
                      OpType _type,
                      const std::vector<int>& axes,
                      bool keepdims);
  TensorHandle reduce_argmax(const TensorHandle _input,
                             const std::vector<int>& axes,
                             bool keepdims);
  TensorHandle reduce_argmin(const TensorHandle _input,
                             const std::vector<int>& axes,
                             bool keepdims);
  TensorHandle reduce_max(const TensorHandle _input,
                          const std::vector<int>& axes,
                          bool keepdims);
  TensorHandle reduce_mean(const TensorHandle _input,
                           const std::vector<int>& axes,
                           bool keepdims);
  TensorHandle reduce_min(const TensorHandle _input,
                          const std::vector<int>& axes,
                          bool keepdims);
  TensorHandle reduce_prod(const TensorHandle _input,
                           const std::vector<int>& axes,
                           bool keepdims);
  TensorHandle reduce_sum(const TensorHandle _input,
                          const std::vector<int>& axes,
                          bool keepdims);
  TensorHandle relu(const TensorHandle _input,
                    bool _inPlace = true);
  TensorHandle reshape(const TensorHandle _input,
                       const std::vector<int>& _shape);
  TensorHandle resize(const TensorHandle _input,
                      const std::vector<int>& _shape);
  TensorHandle round(const TensorHandle _input);
  TensorHandle shape(const TensorHandle _input,
                     OpType _type);
  TensorHandle slice(const TensorHandle _input,
                     const std::vector<int>& _start,
                     const std::vector<int>& _end,
                     const std::vector<int>& _axes,
                     const std::vector<int>& _steps);
  TensorHandle sigmoid(const TensorHandle _input,
                       bool _inPlace = true);
  //void split(Tensor _input, int axis, int c1, int c2, Tensor* outputs);
  //void split(Tensor _input, int axis, int num, const int* sizes, Tensor* outputs);
  void split(const TensorHandle _input, int _axis,
             const std::vector<int>& _sizes,
             TensorHandle* _outputs);
  void split_equal(const TensorHandle _input, int _axis,
                   int _num, TensorHandle* _outputs);
  TensorHandle sqrt(const TensorHandle _input);
  TensorHandle squeeze(const TensorHandle input, const std::vector<int>& axes);
  TensorHandle transpose(const TensorHandle _input,
                         const std::vector<int>& _perm,
                         bool _shuffle = false);
  TensorHandle tanh(const TensorHandle _input,
                    bool _inPlace = true);
  void topk(const TensorHandle _input,
            int _axis, int _numk,
            bool _largest, bool _sorted,
            Tensor* outputs);
  TensorHandle unsqueeze(const TensorHandle input, const std::vector<int>& axes);
  TensorHandle where(const TensorHandle _cond, const TensorHandle _x, const TensorHandle _y);
  //void split(Tensor _input, int axis, int num, Tensor* outputs);

  // Helper Functions for Cython
  Op find_op_or_fail(size_t guid);
  Graph* optimize(float alpha, int budget, bool print_subst);
  Graph* preprocess_weights(void);
  int get_operator_list(Op* opList, size_t maxNumOps);
  int get_input_edges(Edge* opList, size_t guid);
  OpType get_operator_type(size_t guid);
  int get_operator_int_attr(size_t guid, PMParameter attr);
  int get_num_outputs(size_t guid);
  int get_input_dims(size_t guid, int* dims, int idx);
  void get_weight_value(size_t guid, DATATYPE* value);
  int get_output_dims(size_t guid, int* dims, int idx);
  int get_split_lens(size_t guid, int* lens);
  size_t num_in_edges(Op op);
  size_t num_out_edges(Op op);
  size_t hash(void);
  void print(void);
  bool check_correctness(void);
  bool has_loop(void);
  float total_cost(void);
  float run();
  void print_costs(void);
  void print_measurements(void);
#ifdef TRT
  void buildTRTNetwork(INetworkDefinition *network);
private:
  void buildTRTNetworkHelper(INetworkDefinition *network, std::map<SrcEdge, ITensor *, SrcEdgeCompare>& outputs, Edge edge);
#endif
  void export_op(ofstream &file_stream, Op &op);
private:
  TensorHandle input_wrapper(const TensorHandle _input);
  TensorHandle weight_wrapper(const TensorHandle _weight);
public:
  Model *model;
  float totalCost;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare> inEdges, outEdges;
  struct GraphSubst {
    std::vector<Op> srcOps, dstOps;
  };
  std::vector<GraphSubst> subst_history;
};

class Constant : public OpBase {
public:
  Constant(Model* _model, int ndim, int* dims, OpType _type);
  ~Constant(void);
  void forward(bool block);
  void map(void);
  void unmap(void);
  bool get_int_parameter(PMParameter para, int*);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
};

class Conv2D : public OpBase {
public:
  Conv2D(Model* _model, Tensor _input, Tensor _weight,
         int _strideH, int _strideW,
         PaddingMode _padding,
         ActiMode _activation);
  ~Conv2D(void);
  void forward(bool block);
  void map(void);
  void unmap(void);
  bool get_int_parameter(PMParameter para, int*);
  void get_padding(int* padH, int* padW);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
#ifdef USE_CUDNN
  cudnnConvolutionFwdAlgo_t selectForwardAlgorithm(void);
#endif
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
#endif
  int strideH, strideW;
  PaddingMode padding;
  ActiMode activation;
  void *biasPtr;
};

class Matmul : public OpBase {
public:
  Matmul(Model* _model, Tensor _input, Tensor _weight,
         ActiMode _actiMode);
  ~Matmul(void);
  void forward(bool block);
  void map(void);
  void unmap(void);
  bool get_int_parameter(PMParameter para, int*);
  void set_layout(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  int outputC;
  ActiMode activation;
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
#endif
};

class Mul : public OpBase {
public:
  Mul(Model* _model, const Tensor& x, const Tensor& y);
  ~Mul(void);
  void forward(bool block);
  void map(void);
  void unmap(void);
  bool get_int_parameter(PMParameter para, int*);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
};

class Pool2D : public OpBase {
public:
  Pool2D(Model* _model, Tensor _input,
         Tensor _weight, OpType _type,
         int _kernelH, int _kernelW,
         int _strideH, int _strideW,
         PaddingMode _padding, ActiMode _activation);
  ~Pool2D(void);
  bool get_int_parameter(PMParameter para, int*);
  void get_padding(int* padH, int* padW);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
#endif
  int kernelH, kernelW, strideH, strideW;
  PaddingMode padding;
  ActiMode activation;
};

class Activation : public OpBase {
public:
  Activation(Model* _model, Tensor _input, OpType _type, bool _inPlace);
  ~Activation(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor;
  cudnnActivationDescriptor_t actiDesc;
#endif
  bool inPlace;
};

class BatchNorm : public OpBase {
public:
  BatchNorm(Model* _model, Tensor _input, Tensor _scale,
            Tensor _bias, Tensor _mean, Tensor _var);
  ~BatchNorm(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
#endif
  //DATATYPE *biasPtr, *scalePtr, *runningMean, *runningVar, *saveMean, *saveVar;
};

class Cast : public OpBase {
public:
  Cast(Model* _model, const Tensor& _input, DataType _datatype);
  ~Cast(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
};

class Concat : public OpBase {
public:
  Concat(Model* _model, int _axis, int _n, Tensor* _inputs, bool* _needCopy);
  ~Concat(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  int axis;
  bool needCopy[MAX_NUM_INPUTS];
};

class Element : public OpBase {
public:
  Element(Model* _model, OpType _type, const Tensor& _t1, const Tensor& _t2);
  ~Element(void);
  bool use_cudnn_kernel(void) const;
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t in1Tensor, in2Tensor, outTensor;
  cudnnOpTensorDescriptor_t opDesc;
#endif
};

class ElementWiseUnary : public OpBase {
public:
  ElementWiseUnary(Model* _model, const Tensor& _input, OpType _type);
  ~ElementWiseUnary(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
};

class Enlarge : public OpBase {
public:
  Enlarge(Model* _model, Tensor _w1, Tensor _w2);
  ~Enlarge(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
};

class TopK : public OpBase {
public:
  TopK(Model* _model, const Tensor& _input,
       int _axis, int _numk,
       bool _largest, bool _sorted);
  ~TopK(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  int axis;
  bool largest, sorted;
};

class MergeGConv : public OpBase {
public:
  MergeGConv(Model* _model, const Tensor& _weight, int count);
  ~MergeGConv(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  int count;
};

class NoOp : public OpBase {
public:
  NoOp(Model* _model, Tensor _input, OpType _type);
  ~NoOp(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
};

class Pad : public OpBase {
public:
  Pad(Model* _model, const Tensor& _input,
      const std::vector<int>& _pad_before,
      const std::vector<int>& _pad_after,
      float _pad_value);
  ~Pad(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  std::vector<int> pad_before, pad_after;
  float pad_value;
};

class Reduce : public OpBase {
public:
  Reduce(Model* _model, const Tensor& _input, OpType _type,
         const std::vector<int>& _axes, bool _keepdims);
  ~Reduce(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  bool keepdims;
  std::vector<int> axes;
};

class Reshape : public OpBase {
public:
  Reshape(Model* _model, Tensor _input, const std::vector<int>& shape);
  ~Reshape(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
};

class Resize : public OpBase {
public:
  Resize(Model* _model, const Tensor& _input, const std::vector<int>& _shape);
  ~Resize(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  std::vector<int> shape;
};

class Shape : public OpBase {
public:
  Shape(Model* _model, const Tensor& _input, OpType _type);
  ~Shape(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
};

class Slice : public OpBase {
public:
  Slice(Model* _model, const Tensor& _input,
        const std::vector<int>& _start,
        const std::vector<int>& _end,
        const std::vector<int>& _axes,
        const std::vector<int>& _steps);
  ~Slice(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  std::vector<int> start, end, axes, steps;
};

class Split : public OpBase {
public:
  Split(Model* _model, const Tensor& _input, int axis, const std::vector<int>& _sizes);
  ~Split(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  int axis;
  std::vector<int> sizes;
};

class Squeeze : public OpBase {
public:
  Squeeze(Model* _model, const Tensor& input, const std::vector<int>& axes);
  ~Squeeze(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  std::vector<int> axes;
};

class Transpose : public OpBase {
public:
  Transpose(Model* _model, Tensor _input,
            const std::vector<int>& perm,
            bool _shuffle);
  ~Transpose(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  int permIdx;
  bool shuffle;
};

class Unsqueeze : public OpBase {
public:
  Unsqueeze(Model* _model, const Tensor& input, const std::vector<int>& axes);
  ~Unsqueeze(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  std::vector<int> axes;
};

class Where : public OpBase {
public:
  Where(Model* _model, const Tensor& _input, const Tensor& _x, const Tensor& _y);
  ~Where(void);
  bool get_int_parameter(PMParameter para, int*);
  void forward(bool block);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
};

template<typename T>
struct KeyCompare {
  bool operator()(const T& a, const T& b) const {
    for (int i = 0; i < T::KEY_LENGTH; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

struct ActivationKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 2;
  ActivationKey(Tensor, OpType, bool);
  int keys[KEY_LENGTH];
};

// key is (inputN, inputC, inputH, inputW)
struct BatchNormKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH;
  BatchNormKey(Tensor);
  int keys[KEY_LENGTH];
};

struct CastKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 1;
  CastKey(const Tensor& _input, DataType _datatype);
  int keys[KEY_LENGTH];
};

struct ConcatKey {
  static const int KEY_LENGTH = MAX_NUM_INPUTS * Tensor::MAX_KEY_LENGTH + 3;
  ConcatKey(int, int, Tensor*, bool*);
  int keys[KEY_LENGTH];
};

//keys are (ndim, dims[0..ndims-1], constant_mode
struct ConstantKey {
  static const int KEY_LENGTH = MAX_DIM + 2;
  ConstantKey(int, int*, OpType);
  int keys[KEY_LENGTH];
};

// keys are (strideH, strideW, padding, activation, input, weight)
struct Conv2DKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH * 2 + 4;
  Conv2DKey(Tensor, Tensor, int, int,
            PaddingMode, ActiMode);
  int keys[KEY_LENGTH];
};

struct ElementKey {
  static const int KEY_LENGTH = 2*Tensor::MAX_KEY_LENGTH + 1;
  ElementKey(const Tensor& t1, const Tensor& t2, OpType type);
  int keys[KEY_LENGTH];
};

struct ElementWiseUnaryKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 1;
  ElementWiseUnaryKey(const Tensor& _input, OpType _type);
  int keys[KEY_LENGTH];
};

struct EnlargeKey {
  static const int KEY_LENGTH = 2 * Tensor::MAX_KEY_LENGTH;
  EnlargeKey(Tensor w1, Tensor w2);
  int keys[KEY_LENGTH];
};

struct TopKKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 4;
  TopKKey(const Tensor& _input, int _axis, int _numk, bool _largest, bool _sorted);
  int keys[KEY_LENGTH];
};

// keys are (inputX, inputN, inputC, outputC, acti)
//
struct MatmulKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH * 2 + 1;
  MatmulKey(Tensor, Tensor, ActiMode);
  int keys[KEY_LENGTH];
};

struct MergeGConvKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 1;
  MergeGConvKey(const Tensor& weight, int count);
  int keys[KEY_LENGTH];
};

// keys are (inputX, inputN, inputC, outputC, acti)
struct MulKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH * 2;
  MulKey(const Tensor&, const Tensor&);
  int keys[KEY_LENGTH];
};

struct NoopKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 1;
  NoopKey(Tensor input, OpType typee);
  int keys[KEY_LENGTH];
};

struct PadKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 2 * MAX_DIM + 1;
  PadKey(const Tensor& _input,
         const std::vector<int>& _pad_before,
         const std::vector<int>& _pad_after,
         float _pad_value);
  int keys[KEY_LENGTH];
};

// keys are (inputN, inputC, inputH, inputW, kernelH, kernelW,              
//           strideH, strideW, padding, activation, type,
//           input.split[0], input.split[1]
struct Pool2DKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 7;
  Pool2DKey(Tensor, OpType, int, int, int, int,
            PaddingMode, ActiMode);
  int keys[KEY_LENGTH];
};

struct ReduceKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_DIM + 3;
  ReduceKey(const Tensor&, OpType, const std::vector<int>&, bool);
  int keys[KEY_LENGTH];
};

struct ReshapeKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_DIM + 1;
  ReshapeKey(Tensor, const std::vector<int>&);
  int keys[KEY_LENGTH];
};

struct ResizeKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_DIM + 1;
  ResizeKey(const Tensor&, const std::vector<int>&);
  int keys[KEY_LENGTH];
};

struct ShapeKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 1;
  ShapeKey(const Tensor& _input, OpType _type);
  int keys[KEY_LENGTH];
};

struct SliceKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_DIM * 4 + 1;
  SliceKey(const Tensor& _input,
           const std::vector<int>& _start,
           const std::vector<int>& _end,
           const std::vector<int>& _axes,
           const std::vector<int>& _steps);
  int keys[KEY_LENGTH];
};

struct SqueezeKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_DIM;
  SqueezeKey(const Tensor& input, const std::vector<int>& axes);
  int keys[KEY_LENGTH];
};

struct SplitKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_NUM_OUTPUTS + 2;
  SplitKey(const Tensor& _input, int _axis, const std::vector<int>& _sizes);
  int keys[KEY_LENGTH];
};

struct TransposeKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + 2;
  TransposeKey(Tensor, const std::vector<int>&, bool);
  int keys[KEY_LENGTH];
};

struct UnsqueezeKey {
  static const int KEY_LENGTH = Tensor::MAX_KEY_LENGTH + MAX_DIM;
  UnsqueezeKey(const Tensor& input, const std::vector<int>& axes);
  int keys[KEY_LENGTH];
};

struct WhereKey {
  static const int KEY_LENGTH = 3 * Tensor::MAX_KEY_LENGTH;
  WhereKey(const Tensor& _cond, const Tensor& _x, const Tensor& _y);
  int keys[KEY_LENGTH];
};

class Model {
public:
  Model();
  Op get_or_create_activation(Tensor _input, OpType _type,
                              bool _inPlace);
  Op get_or_create_batchnorm(Tensor _input, Tensor _scale, Tensor _bias,
                             Tensor _mean, Tensor _var);
  Op get_or_create_cast(const Tensor& _input, DataType _datatype);
  Op get_or_create_concat(int axis, int n, Tensor* _inputs, bool* _needCopy);
  Op get_or_create_constant(int ndim, int* dims, OpType type);
  Op get_or_create_conv2d(Tensor _input, Tensor _weight,
                          int _strideH, int _strideW,
                          PaddingMode _padding,
                          ActiMode _activation);
  Op get_or_create_element(OpType type, const Tensor& t1, const Tensor& t2);
  Op get_or_create_elementwise_unary(const Tensor& _input, OpType _type);
  Op get_or_create_enlarge(Tensor _w1, Tensor _w2);
  Op get_or_create_matmul(Tensor _input, Tensor _weight,
                          ActiMode _actimode);
  Op get_or_create_mul(const Tensor& x,
                       const Tensor& y);
  Op get_or_create_pad(const Tensor& _input,
                       const std::vector<int>& _pad_before,
                       const std::vector<int>& _pad_after,
                       float _pad_value);
  Op get_or_create_pool2d(Tensor _input, Tensor _weight,
                          OpType _type,
                          int _kernelH, int _kernelW,
                          int _strideH, int _strideW,
                          PaddingMode _padding,
                          ActiMode _activation);
  Op get_or_create_reduce(const Tensor& _input, OpType _type,
                          const std::vector<int>& _axes, bool _keepdims);
  Op get_or_create_reshape(Tensor _input, const std::vector<int>& shape);
  Op get_or_create_resize(const Tensor& _input,
                          const std::vector<int>& _shape);
  Op get_or_create_shape(const Tensor& _input, OpType _type);
  Op get_or_create_slice(const Tensor& _input,
                         const std::vector<int>& _start,
                         const std::vector<int>& _end,
                         const std::vector<int>& _axes,
                         const std::vector<int>& _steps);
  Op get_or_create_squeeze(const Tensor& input, const std::vector<int>& axes);
  Op get_or_create_split(const Tensor& _input, int _axis, const std::vector<int>& _sizes);
  Op get_or_create_split(const Tensor& _input, int axis, int n);
  Op get_or_create_topk(const Tensor& _input, int _axis, int _numk,
                        bool _largest, bool _sorted);
  Op get_or_create_transpose(Tensor _input, const std::vector<int>& _perm,
                             bool _shuffle);
  Op get_or_create_transpose(Tensor _input, int permIdx,
                             bool _shuffle);
  Op get_or_create_noop(Tensor _input, OpType _type);
  Op get_or_create_merge_gconv(const Tensor& _weight,
                               int count);
  Op get_or_create_unsqueeze(const Tensor& input, const std::vector<int>& axes);
  Op get_or_create_where(const Tensor& _cond, const Tensor& _x, const Tensor& _y);
  // Special API for creating weight and input operator
  Op create_input(Tensor _input, OpType _type);
  Op create_weight(Tensor _weight, OpType _type);
  void measure_conv2d_cost(Conv2D*);
  void measure_matmul_cost(Matmul*);
  void measure_mul_cost(Mul*);
  void measure_pad_cost(Pad*);
  void measure_pool2d_cost(Pool2D*);
  void measure_topk_cost(TopK*);
  void measure_transpose_cost(Transpose*);
  void measure_reduce_cost(Reduce*);
  void measure_reshape_cost(Reshape*);
  void measure_resize_cost(Resize*);
  void measure_activation_cost(Activation*);
  void measure_batchnorm_cost(BatchNorm*);
  void measure_cast_cost(Cast*);
  void measure_concat_cost(Concat*);
  void measure_shape_cost(Shape*);
  void measure_slice_cost(Slice*);
  void measure_split_cost(Split*);
  void measure_element_cost(Element*);
  void measure_elementwise_unary_cost(ElementWiseUnary*);
  void measure_enlarge_cost(Enlarge*);
  void measure_squeeze_cost(Squeeze*);
  void measure_unsqueeze_cost(Unsqueeze*);
  void measure_where_cost(Where*);
  void* allocate_memory(size_t size, const DATATYPE* initial_data= NULL);
  bool copy_memory(DATATYPE* dst, const DATATYPE* src, size_t size);
  float measure_oplist_runtime(const std::vector<OpBase*>& list);
  bool broadcastable(const Tensor& t1, const Tensor& t2);
public:
  bool isTraining;
  bool print_cost;
  size_t global_unique_id;
  size_t workSpaceSize;
  void* workSpace;
#ifdef USE_CUDNN
  cudnnHandle_t dnn;
  cublasHandle_t blas;
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  // Note that actiDesc is set when we construct Model since
  // all relus are identical.
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudaEvent_t startEvent, endEvent;
  // variables for batch norm
  cudnnTensorDescriptor_t scaleTensor;
  // variables for element wise
  cudnnOpTensorDescriptor_t opDesc;
#endif
#ifdef USE_DNNL
  DNNLNet net;
  dnnl::engine eng;
  dnnl::stream strm;
#endif
  std::map<ActivationKey, Activation*, KeyCompare<ActivationKey> > activation;
  std::map<BatchNormKey, BatchNorm*, KeyCompare<BatchNormKey> > batchnorm;
  std::map<CastKey, Cast*, KeyCompare<CastKey> > cast;
  std::map<ConcatKey, Concat*, KeyCompare<ConcatKey> > concat;
  std::map<ConstantKey, Constant*, KeyCompare<ConstantKey> > constant;
  std::map<Conv2DKey, Conv2D*, KeyCompare<Conv2DKey> > conv2d;
  std::map<ElementKey, Element*, KeyCompare<ElementKey> > element;
  std::map<ElementWiseUnaryKey, ElementWiseUnary*, KeyCompare<ElementWiseUnaryKey> > element_unary;
  std::map<EnlargeKey, Enlarge*, KeyCompare<EnlargeKey> > enlarge;
  std::map<MatmulKey, Matmul*, KeyCompare<MatmulKey> > matmul;
  std::map<MergeGConvKey, MergeGConv*, KeyCompare<MergeGConvKey> > merge_gconv;
  std::map<MulKey, Mul*, KeyCompare<MulKey> > mul;
  std::map<NoopKey, NoOp*, KeyCompare<NoopKey> > noop;
  std::map<PadKey, Pad*, KeyCompare<PadKey> > pad;
  std::map<Pool2DKey, Pool2D*, KeyCompare<Pool2DKey> > pool2d;
  std::map<ReduceKey, Reduce*, KeyCompare<ReduceKey> > reduce;
  std::map<ReshapeKey, Reshape*, KeyCompare<ReshapeKey> > reshape;
  std::map<ResizeKey, Resize*, KeyCompare<ResizeKey> > resize;
  std::map<ShapeKey, Shape*, KeyCompare<ShapeKey> > shape;
  std::map<SliceKey, Slice*, KeyCompare<SliceKey> > slice;
  std::map<SplitKey, Split*, KeyCompare<SplitKey> > split;
  std::map<SqueezeKey, Squeeze*, KeyCompare<SqueezeKey> > squeeze;
  std::map<TopKKey, TopK*, KeyCompare<TopKKey> > topk;
  std::map<TransposeKey, Transpose*, KeyCompare<TransposeKey> > transpose;
  std::map<UnsqueezeKey, Unsqueeze*, KeyCompare<UnsqueezeKey> > unsqueeze;
  std::map<WhereKey, Where*, KeyCompare<WhereKey> > where;
  DATATYPE *inputPtr, *biasPtr, *outputPtr, *filterPtr;
  // variables for batch norm
  DATATYPE *scalePtr, *runningMean, *runningVar, *saveMean, *saveVar;
};

} // namespace taso
#endif
