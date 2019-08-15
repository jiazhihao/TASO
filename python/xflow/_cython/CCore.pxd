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

cdef extern from "xflow/ops.h" namespace "XFlow":
    # This must be consistent with include/xflow/ops.h
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

    # This must be consistent with include/xflow/ops.h
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

    # This must be consistent with include/xflow/ops.h
    cdef enum ActiMode:
        AC_MODE_NONE
        AC_MODE_SIGMOID
        AC_MODE_RELU
        AC_MODE_TANH

    # This must be consistent with include/xflow/ops.h
    cdef enum PaddingMode:
        PD_MODE_SAME
        PD_MODE_VALID

    # This must be consistent with include/xflow/ops.h
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
        TensorHandle reshape(const TensorHandle input,
                             const vector[int] shape)
        TensorHandle relu(const TensorHandle input,
                          bool _inplace)
        TensorHandle sigmoid(const TensorHandle input,
                            bool _inplace)
        TensorHandle tanh(const TensorHandle input,
                          bool _inplace)
        TensorHandle transpose(const TensorHandle input,
                               const vector[int] perm,
                               bool shuffle)
        TensorHandle new_input(int ndim, const int* dims)
        TensorHandle new_weight(int ndim, const int* dims, const float* data)
        Graph* optimize(float alpha, int budget)
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
        float run()
