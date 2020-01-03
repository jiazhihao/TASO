# Copyright 2019 Stanford
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#ccore.pxd

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "taso/ops.h" namespace "taso":
    # This must be consistent with include/taso/ops.h
    cdef enum DataType:
        DT_FLOAT  = 111,
        DT_DOUBLE = 222,
        DT_HALF   = 333,
        DT_INT8   = 444,
        DT_UINT8  = 555,
        DT_INT32  = 666,
        DT_INT64  = 777,
        DT_BOOL   = 888,

    # This must be consistent with include/taso/ops.h
    cdef enum OpType:
        OP_INPUT
        OP_WEIGHT
        OP_ANY
        OP_CONV2D
        OP_DROPOUT
        OP_LINEAR
        OP_POOL2D_MAX
        OP_POOL2D_AVG
        OP_RELU
        OP_SIGMOID
        OP_TANH
        OP_BATCHNORM
        OP_CONCAT
        OP_SPLIT
        OP_RESHAPE
        OP_TRANSPOSE
        # RNN operators
        OP_EW_ADD
        OP_EW_MUL
        OP_MATMUL
        OP_SCALARMUL
        OP_ENLARGE
        OP_MERGE_GCONV
        OP_CONSTANT_IMM,
        OP_CONSTANT_ICONV,
        OP_CONSTANT_ONE,
        OP_CONSTANT_POOL,
        OP_SQUEEZE,
        OP_UNSQUEEZE,
        OP_EW_SUB,
        OP_EW_DIV,
        OP_EW_EQUAL,
        OP_EW_GREATER,
        OP_EW_LESS,
        OP_EW_MAX,
        OP_EW_MIN,
        OP_REDUCE_ARGMAX,
        OP_REDUCE_ARGMIN,
        OP_REDUCE_MAX,
        OP_REDUCE_MEAN,
        OP_REDUCE_MIN,
        OP_REDUCE_PROD,
        OP_REDUCE_SUM,
        OP_PAD,
        OP_SHAPE,
        OP_SIZE,
        OP_TOPK,
        OP_WHERE,
        OP_CEIL,
        OP_CAST,
        OP_EXP,
        OP_ROUND,
        OP_LOG,
        OP_LOGICAL_NOT,
        OP_SQRT,
        OP_LEAKYRELU,
        OP_SLICE,
        OP_RESIZE,
        OP_PRELU,
        OP_FUSE_CONV_BATCHNORM,

    # This must be consistent with include/taso/ops.h
    cdef enum PMParameter:
        PM_OP_TYPE
        PM_NUM_INPUTS
        PM_NUM_OUTPUTS
        PM_GROUP
        PM_KERNEL_H
        PM_KERNEL_W
        PM_STRIDE_H
        PM_STRIDE_W
        PM_PAD
        PM_ACTI
        PM_NUMDIM
        PM_AXIS
        PM_PERM
        PM_OUTSHUFFLE
        PM_MERGE_GCONV_COUNT
        PM_AXES

    # This must be consistent with include/taso/ops.h
    cdef enum ActiMode:
        AC_MODE_NONE
        AC_MODE_SIGMOID
        AC_MODE_RELU
        AC_MODE_TANH

    # This must be consistent with include/taso/ops.h
    cdef enum PaddingMode:
        PD_MODE_SAME
        PD_MODE_VALID

    # This must be consistent with include/taso/ops.h
    cdef enum ConstantMode:
        CN_MODE_IDENTITY
        CN_MODE_ZEROS
        CN_MODE_ONES
        CN_MODE_ONES_SCALED_L1
        CN_MODE_ONES_SCALED_L2
        CN_MODE_ONES_SCALED_ALL

    cdef cppclass Model:
        Model()

#    ctypedef struct SplitInfo:
#        int num
#        int pos[MAX_NUM_SPLITS]
#
#    ctypedef cppclass OpBase:
#        pass

    ctypedef struct Op:
        size_t guid
        pass

    ctypedef struct Edge:
        Op srcOp
        Op dstOp
        int srcIdx
        int dstIdx    

    ctypedef struct Tensor:
        int numDim
        int dim[4]
        int stride[4] # NOTE: this must be consistent with the C++ header
        pass
#        int idx
#        Op op
#        void* ptr
#        SplitInfo split[MAX_DIM]

    ctypedef Tensor* TensorHandle

    cdef cppclass Graph:
        Graph()
        TensorHandle batchnorm(const TensorHandle input,
                               const TensorHandle scale,
                               const TensorHandle bias,
                               const TensorHandle mean,
                               const TensorHandle var)
        TensorHandle cast(const TensorHandle input, DataType datatype)
        TensorHandle ceil(const TensorHandle input)
        TensorHandle concat(int axis, int n,
                            const TensorHandle* inputs)
        TensorHandle conv2d(const TensorHandle input,
                            const TensorHandle weight,
                            int strideH, int strideW,
                            PaddingMode _padding,
                            ActiMode _activation)
        TensorHandle dropout(const TensorHandle input)
        TensorHandle element(OpType type,
                             const TensorHandle x,
                             const TensorHandle y)
        TensorHandle exp(const TensorHandle input)
        TensorHandle log(const TensorHandle input)
        TensorHandle logical_not(const TensorHandle input)
        TensorHandle pool2d_max(const TensorHandle input,
                                int kernelH, int kernelW,
                                int strideH, int strideW,
                                PaddingMode padding,
                                ActiMode activation)
        TensorHandle pool2d_avg(const TensorHandle input,
                                int kernelH, int kernelW,
                                int strideH, int strideW,
                                PaddingMode padding,
                                ActiMode activation)
        TensorHandle matmul(const TensorHandle input,
                            const TensorHandle weight,
                            ActiMode activation)
        TensorHandle reduce_argmax(const TensorHandle input,
                                   const vector[int] axes,
                                   bool keepdims)
        TensorHandle reduce_argmin(const TensorHandle input,
                                   const vector[int] axes,
                                   bool keepdims)
        TensorHandle reduce_max(const TensorHandle input,
                                const vector[int] axes,
                                bool keepdims)
        TensorHandle reduce_mean(const TensorHandle input,
                                 const vector[int] axes,
                                 bool keepdims)
        TensorHandle reduce_min(const TensorHandle input,
                                const vector[int] axes,
                                bool keepdims)
        TensorHandle reduce_prod(const TensorHandle input,
                                 const vector[int] axes,
                                 bool keepdims)
        TensorHandle reduce_sum(const TensorHandle input,
                                const vector[int] axes,
                                bool keepdims)
        TensorHandle reshape(const TensorHandle input,
                             const vector[int] shape)
        TensorHandle leakyrelu(const TensorHandle input,
                               float alpha, bool _inplace)
        TensorHandle relu(const TensorHandle input,
                          bool _inplace)
        TensorHandle round(const TensorHandle input)
        TensorHandle shape(const TensorHandle input, OpType type)
        TensorHandle sigmoid(const TensorHandle input,
                            bool _inplace)
        TensorHandle slice(const TensorHandle input,
                           const vector[int] start,
                           const vector[int] end,
                           const vector[int] axse,
                           const vector[int] steps)
        void split(const TensorHandle input, int axis,
                   const vector[int] sizes, TensorHandle* outputs)
        void split_equal(const TensorHandle input, int axis,
                         int num, TensorHandle* outputs)
        TensorHandle sqrt(const TensorHandle input)
        TensorHandle squeeze(const TensorHandle input,
                              const vector[int] axes)
        TensorHandle tanh(const TensorHandle input,
                          bool _inplace)
        TensorHandle transpose(const TensorHandle input,
                               const vector[int] perm,
                               bool shuffle)
        TensorHandle unsqueeze(const TensorHandle input,
                               const vector[int] axes)
        TensorHandle new_input(int ndim, const int* dims)
        TensorHandle new_weight(int ndim, const int* dims, const float* data)
        Graph* optimize(float alpha, int budget, bool print_subst)
        int get_operator_list(Op* ops, size_t maxNumOps)
        int get_input_edges(Edge* edges, size_t guid)
        OpType get_operator_type(size_t guid)
        int get_operator_int_attr(size_t guid, PMParameter attr)
        int get_num_outputs(size_t guid)
        int get_input_dims(size_t guid, int* dims, int idx)
        void get_weight_value(size_t guid, float* data)
        int get_split_lens(size_t guid, int* lens)
        int get_output_dims(size_t guid, int* dims, int idx)
        void print_measurements()
        float total_cost()
        float run()
