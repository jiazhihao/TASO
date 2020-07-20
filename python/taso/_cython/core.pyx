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

from CCore cimport Model
from CCore cimport Graph
from CCore cimport Tensor
from CCore cimport *
from cpython cimport array
import ctypes
import array
import numpy as np

#helper function
def get_padding_mode(padding):
    if (padding == "SAME"):
        return PD_MODE_SAME
    elif (padding == "VALID"):
        return PD_MODE_VALID
    else:
        assert(False)

def get_data_type(datatype):
    if datatype == "FLOAT":
        return DT_FLOAT
    elif datatype == "DOUBLE":
        return DT_DOUBLE
    elif datatype == "FLOAT16":
        return DT_HALF
    elif datatype == "INT8":
        return DT_INT8
    elif datatype == "UINT8":
        return DT_UINT8
    elif datatype == "INT32":
        return DT_INT32
    elif datatype == "INT64":
        return DT_INT64
    elif datatype == "BOOL":
        return DT_BOOL

def get_activation_mode(activation):
    if (activation == "NONE"):
        return AC_MODE_NONE
    elif (activation == "SIGMOID"):
        return AC_MODE_SIGMOID
    elif (activation == "RELU"):
        return AC_MODE_RELU
    elif (activation == "TANH"):
        return AC_MODE_TANH
    else:
        assert(False)

cdef class PyModel:
    cdef Model *p_model # Hold a Model instance

    def __cinit__(self):
        self.p_model = new Model()

    def __dealloc__(self):
        del self.p_model

cdef class PyTensor:
    cdef TensorHandle ctensor # Hold a Tensor instance

    cdef inline _set_tensor(self, tensor):
        cdef unsigned long long ptr
        if tensor is None:
            self.ctensor = <TensorHandle>(NULL)
        else:
            ptr = ctypes.cast(tensor, ctypes.c_void_p).value
            self.ctensor = <TensorHandle>(ptr)

    property tensor:
        def __get__(self):
            if self.ctensor == NULL:
                return None
            else:
                return ctypes.cast(<unsigned long long>self.ctensor, ctypes.c_void_p)
        
        def __set__(self, value):
            self._set_tensor(value)

    property nDim:
        def __get__(self):
            if self.ctensor == NULL:
                return None
            else:
                return self.ctensor.numDim

    def __cinit__(self, tensor):
        self._set_tensor(tensor)

    def dim(self, int idx):
        if (idx < self.ctensor.numDim):
            return self.ctensor.dim[idx]
        else:
            assert False , "Error: index out of range"
            return None

# Construct operator table
op_table = dict()
op_table[OP_INPUT] = "Input"
op_table[OP_WEIGHT] = "Weight"
op_table[OP_CONV2D] = "Conv"
op_table[OP_DROPOUT] = "Dropout"
op_table[OP_POOL2D_MAX] = "MaxPool"
op_table[OP_POOL2D_AVG] = "AveragePool"
op_table[OP_RELU] = "Relu"
op_table[OP_SIGMOID] = "Sigmoid"
op_table[OP_TANH] = "Tanh"
op_table[OP_BATCHNORM] = "BatchNormalization"
op_table[OP_CONCAT] = "Concat"
op_table[OP_SPLIT] = "Split"
op_table[OP_RESHAPE] = "Reshape"
op_table[OP_TRANSPOSE] = "Transpose"
op_table[OP_EW_ADD] = "Add"
op_table[OP_EW_MUL] = "Mul"
op_table[OP_MATMUL] = "MatMul"
op_table[OP_SQUEEZE] = "Squeeze"
op_table[OP_UNSQUEEZE] = "Unsqueeze"
op_table[OP_EW_SUB] = "Sub"
op_table[OP_EW_DIV] = "Div"
op_table[OP_EW_EQUAL] = "Equal"
op_table[OP_EW_GREATER] = "Greater"
op_table[OP_EW_LESS] = "Less"
op_table[OP_EW_MAX] = "Max"
op_table[OP_EW_MIN] = "Min"
op_table[OP_REDUCE_ARGMAX] = "ArgMax"
op_table[OP_REDUCE_ARGMIN] = "ArgMin"
op_table[OP_REDUCE_MAX] = "ReduceMax"
op_table[OP_REDUCE_MEAN] = "ReduceMean"
op_table[OP_REDUCE_MIN] = "ReduceMin"
op_table[OP_REDUCE_PROD] = "ReduceProd"
op_table[OP_REDUCE_SUM] = "ReduceSum"
op_table[OP_PAD] = "Pad"
op_table[OP_SHAPE] = "Shape"
op_table[OP_SIZE] = "Size"
op_table[OP_TOPK] = "TopK"
op_table[OP_WHERE] = "Where"
op_table[OP_CEIL] = "Ceil"
op_table[OP_CAST] = "Cast"
op_table[OP_EXP] = "Exp"
op_table[OP_ROUND] = "Round"
op_table[OP_LOG] = "Log"
op_table[OP_LOGICAL_NOT] = "Not"
op_table[OP_SQRT] = "Sqrt"
op_table[OP_SLICE] = "Slice"
op_table[OP_RESIZE] = "Resize"
# op_table[OP_BROADCAST_ADD] = "BroadcastAdd"
op_table[OP_BROADCAST_ADD] = "Add"

cdef class PyGraph:
    cdef Graph *p_graph #Hold a Graph instance

    def __cinit__(self, graph = None):
        cdef unsigned long long ptr
        if graph is None:
            self.p_graph = new Graph()
        else:
            ptr = ctypes.cast(graph, ctypes.c_void_p).value
            self.p_graph = <Graph*>(ptr)
    def print_measurements(self):
        self.p_graph.print_measurements()

    def run_time(self):
        return self.p_graph.run()

    def cost(self):
        return self.p_graph.total_cost()

    #def __dealloc__(self):
        #t = ctypes.cast(<unsigned long long>self.p_graph, ctypes.c_void_p)
        #print(t)
        #del self.p_graph

    # element-wise addition
    def add(self, PyTensor x, PyTensor y):
        cdef TensorHandle handle = self.p_graph.element(OP_EW_ADD, x.ctensor, y.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def batchnorm(self, PyTensor input, PyTensor scale, PyTensor bias, PyTensor mean, PyTensor var):
        cdef TensorHandle handle = self.p_graph.batchnorm(input.ctensor, scale.ctensor, bias.ctensor, mean.ctensor, var.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def cast(self, *, PyTensor input, datatype):
        datatype = get_data_type(datatype)
        cdef TensorHandle handle = self.p_graph.cast(input.ctensor, datatype)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)    

    def ceil(self, *, PyTensor input):
        cdef TensorHandle handle = self.p_graph.ceil(input.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def concat(self, int axis, list inputs):
        cdef TensorHandle cinputs[32]
        cdef unsigned long long ptr
        assert len(inputs) <= 32
        for i in range(len(inputs)):
            assert(type(inputs[i]) == PyTensor)
            assert(inputs[i].tensor is not None)
            ptr = ctypes.cast(inputs[i].tensor, ctypes.c_void_p).value
            cinputs[i] = <TensorHandle>(ptr)
        cdef TensorHandle handle = self.p_graph.concat(axis, len(inputs), cinputs)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def conv2d(self, *, PyTensor input, PyTensor weight, strides, padding, activation = "NONE"):
        assert (type(input) == PyTensor)
        padding = get_padding_mode(padding)
        activation = get_activation_mode(activation)
        cdef TensorHandle handle = self.p_graph.conv2d(input.ctensor, weight.ctensor, strides[0], strides[1], padding, activation)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def div(self, *, PyTensor x, PyTensor y):
        cdef TensorHandle handle = self.p_graph.element(OP_EW_DIV, x.ctensor, y.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def dropout(self, PyTensor input, float rate = 0):
        # We ignore dropout rate for inference
        cdef TensorHandle handle = self.p_graph.dropout(input.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def equal(self, *, PyTensor x, PyTensor y):
        cdef TensorHandle handle = self.p_graph.element(OP_EW_EQUAL, x.ctensor, y.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def exp(self, *, PyTensor input):
        cdef TensorHandle handle = self.p_graph.exp(input.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def greater(self, *, PyTensor x, PyTensor y):
        cdef TensorHandle handle = self.p_graph.element(OP_EW_GREATER, x.ctensor, y.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def identity(self, PyTensor input):
        # We ignore dropout rate for inference
        cdef TensorHandle handle = self.p_graph.dropout(input.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def leakyrelu(self, PyTensor input, float alpha, bool inplace = False):
        cdef TensorHandle handle = self.p_graph.leakyrelu(input.ctensor, alpha, inplace)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def less(self, *, PyTensor x, PyTensor y):
        cdef TensorHandle handle = self.p_graph.element(OP_EW_LESS, x.ctensor, y.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def log(self, *, PyTensor input):
        cdef TensorHandle handle = self.p_graph.log(input.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def logical_not(self, *, PyTensor input):
        cdef TensorHandle handle = self.p_graph.logical_not(input.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def matmul(self, PyTensor input, PyTensor weight, activation = "NONE"):
        assert(type(input) == PyTensor)
        activation = get_activation_mode(activation)
        cdef TensorHandle handle = self.p_graph.matmul(input.ctensor, weight.ctensor, activation)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def max(self, PyTensor x, PyTensor y):
        cdef TensorHandle handle = self.p_graph.element(OP_EW_MAX, x.ctensor, y.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def min(self, PyTensor x, PyTensor y):
        cdef TensorHandle handle = self.p_graph.element(OP_EW_MIN, x.ctensor, y.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    # element-wise multiplication
    def mul(self, PyTensor x, PyTensor y):
        cdef TensorHandle handle = self.p_graph.element(OP_EW_MUL, x.ctensor, y.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def maxpool2d(self, PyTensor input, kernels, strides, padding, activation = "NONE"):
        padding = get_padding_mode(padding)
        activation = get_activation_mode(activation)
        cdef TensorHandle handle = self.p_graph.pool2d_max(input.ctensor, kernels[0], kernels[1], strides[0], strides[1], padding, activation)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def avgpool2d(self, *, PyTensor input, kernels, strides, padding, activation = "NONE"):
        padding = get_padding_mode(padding)
        activation = get_activation_mode(activation)
        cdef TensorHandle handle = self.p_graph.pool2d_avg(input.ctensor, kernels[0], kernels[1], strides[0], strides[1], padding, activation)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def prelu(self, *, PyTensor x, PyTensor slope):
        cdef TensorHandle handle = self.p_graph.element(OP_PRELU, x.ctensor, slope.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def reduce_argmax(self, *, PyTensor input, tuple axes, bool keepdims = True):
        cdef vector[int] caxes
        caxes.resize(len(axes))
        for i in range(len(axes)):
            caxes[i] = axes[i]
        cdef TensorHandle handle = self.p_graph.reduce_argmax(input.ctensor, caxes, keepdims)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def reduce_argmin(self, *, PyTensor input, tuple axes, bool keepdims = True):
        cdef vector[int] caxes
        caxes.resize(len(axes))
        for i in range(len(axes)):
            caxes[i] = axes[i]
        cdef TensorHandle handle = self.p_graph.reduce_argmin(input.ctensor, caxes, keepdims)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def reduce_max(self, *, PyTensor input, tuple axes, bool keepdims = True):
        cdef vector[int] caxes
        caxes.resize(len(axes))
        for i in range(len(axes)):
            caxes[i] = axes[i]
        cdef TensorHandle handle = self.p_graph.reduce_max(input.ctensor, caxes, keepdims)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def reduce_mean(self, *, PyTensor input, tuple axes, bool keepdims = True):
        cdef vector[int] caxes
        caxes.resize(len(axes))
        for i in range(len(axes)):
            caxes[i] = axes[i]
        cdef TensorHandle handle = self.p_graph.reduce_mean(input.ctensor, caxes, keepdims)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def reduce_min(self, *, PyTensor input, tuple axes, bool keepdims = True):
        cdef vector[int] caxes
        caxes.resize(len(axes))
        for i in range(len(axes)):
            caxes[i] = axes[i]
        cdef TensorHandle handle = self.p_graph.reduce_min(input.ctensor, caxes, keepdims)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def reduce_prod(self, *, PyTensor input, tuple axes, bool keepdims = True):
        cdef vector[int] caxes
        caxes.resize(len(axes))
        for i in range(len(axes)):
            caxes[i] = axes[i]
        cdef TensorHandle handle = self.p_graph.reduce_prod(input.ctensor, caxes, keepdims)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def reduce_sum(self, *, PyTensor input, tuple axes, bool keepdims = True):
        cdef vector[int] caxes
        caxes.resize(len(axes))
        for i in range(len(axes)):
            caxes[i] = axes[i]
        cdef TensorHandle handle = self.p_graph.reduce_sum(input.ctensor, caxes, keepdims)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def reshape(self, PyTensor input, tuple shape):
        cdef vector[int] cshape
        cshape.resize(len(shape))
        for i in range(len(shape)):
            cshape[i] = shape[i]
        cdef TensorHandle handle = self.p_graph.reshape(input.ctensor, cshape)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def relu(self, PyTensor input, bool inplace = False):
        cdef TensorHandle handle = self.p_graph.relu(input.ctensor, inplace)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def round(self, PyTensor input):
        cdef TensorHandle handle = self.p_graph.round(input.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def shape(self, PyTensor input):
        cdef TensorHandle handle = self.p_graph.shape(input.ctensor, OP_SHAPE)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def sigmoid(self, PyTensor input, bool inplace = False):
        cdef TensorHandle handle = self.p_graph.sigmoid(input.ctensor, inplace)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def size(self, *, PyTensor input):
        cdef TensorHandle handle = self.p_graph.shape(input.ctensor, OP_SIZE)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def slice(self, PyTensor input, start, end, axes, steps):
        cdef vector[int] cstart
        cdef vector[int] cend
        cdef vector[int] caxes
        cdef vector[int] csteps
        cstart.resize(len(start))
        for i in range(len(start)):
            cstart[i] = start[i]
        cend.resize(len(end))
        for i in range(len(end)):
            cend[i] = end[i]
        if axes: 
            caxes.resize(len(axes))
            for i in range(len(axes)):
                caxes[i] = axes[i]
        else:
            caxes.resize(len(start))
            for i in range(len(start)):
                caxes[i] = i
        if steps:
            csteps.resize(len(steps))
            for i in range(len(steps)):
                csteps[i] = steps[i]
        else:
            csteps.resize(len(start))
            for i in range(len(start)):
                csteps[i] = 1
        cdef TensorHandle handle = self.p_graph.slice(input.ctensor, cstart, cend, caxes, csteps)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def split(self, PyTensor input, int axis, sizes):
        cdef TensorHandle coutputs[32]
        cdef vector[int] csizes
        if type(sizes) is list:
            assert len(sizes) <= 32
            csizes.resize(len(sizes))
            for i in range(len(sizes)):
                csizes[i] = sizes[i]
            self.p_graph.split(input.ctensor, axis, csizes, coutputs)
        else:
            # sizes is an integer
            self.p_graph.split_equal(input.ctensor, axis, sizes, coutputs)
        outputs = list()
        for i in range(len(sizes)):
            t = ctypes.cast(<unsigned long long>coutputs[i], ctypes.c_void_p)
            outputs.append(PyTensor(t))
        return outputs

    def sqrt(self, *, PyTensor input):
        cdef TensorHandle handle = self.p_graph.sqrt(input.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def squeeze(self, *, PyTensor input, tuple axes):
        cdef vector[int] caxes
        caxes.resize(len(axes))
        for i in range(len(axes)):
            caxes[i] = axes[i]
        cdef TensorHandle handle = self.p_graph.squeeze(input.ctensor, caxes)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def sub(self, *, PyTensor x, PyTensor y):
        cdef TensorHandle handle = self.p_graph.element(OP_EW_SUB, x.ctensor, y.ctensor)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def tanh(self, PyTensor input, bool inplace = False):
        cdef TensorHandle handle = self.p_graph.tanh(input.ctensor, inplace)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def transpose(self, PyTensor input, tuple perm, bool shuffle = False):
        cdef vector[int] cperm
        cperm.resize(len(perm))
        for i in range(len(perm)):
            cperm[i] = perm[i]
        cdef TensorHandle handle = self.p_graph.transpose(input.ctensor, cperm, shuffle)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def unsqueeze(self, PyTensor input, tuple axes):
        cdef vector[int] caxes
        caxes.resize(len(axes))
        for i in range(len(axes)):
            caxes[i] = axes[i]
        cdef TensorHandle handle = self.p_graph.unsqueeze(input.ctensor, caxes)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def new_input(self, *, tuple dims):
        cdef int ndim = len(dims)
        cdef int dim_array[16]
        assert (ndim < 16)
        for i in range(0, len(dims)):
            dim_array[i] = dims[i]
        cdef TensorHandle handle = self.p_graph.new_input(ndim, dim_array)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def new_weight(self, *, tuple dims, data = None):
        cdef int ndim = len(dims)
        cdef int dim_array[16]
        cdef array.array arr
        if data is None:
            data = np.random.rand(*dims)
        if isinstance(data, np.ndarray):
            assert dims == data.shape
            arr = array.array('f', data.flatten().tolist())
        else:
            arr = array.array('f', data)
        assert (ndim < 16)
        for i in range(0, len(dims)):
            dim_array[i] = dims[i]
        cdef TensorHandle handle = self.p_graph.new_weight(ndim, dim_array, arr.data.as_floats)
        t = ctypes.cast(<unsigned long long>handle, ctypes.c_void_p)
        return PyTensor(t)

    def optimize(self, float alpha, int budget, bool print_subst):
        cdef Graph* new_graph = self.p_graph.optimize(alpha, budget, print_subst)
        graph = ctypes.cast(<unsigned long long>new_graph, ctypes.c_void_p)
        return PyGraph(graph)

    def get_operator_list(self):
        cdef Op ops[4192]
        cdef int numOps = self.p_graph.get_operator_list(ops, 4192)
        opList = list()
        for i in range(numOps):
            #print(ops[i].guid)
            opList.append(ops[i])
        return opList

    def get_input_edges(self, Op op):
        cdef Edge edges[128];
        cdef int numEdges = self.p_graph.get_input_edges(edges, op.guid)
        inEdges = list()
        for i in range(numEdges):
            inEdges.append(edges[i])
        return inEdges

    def get_input_dims(self, Op op, int idx):
        cdef int dims[8]
        cdef int ndims = self.p_graph.get_input_dims(op.guid, dims, idx)
        dimlist = list()
        for i in range(ndims):
            dimlist.append(dims[i])
        return dimlist

    def get_weight_value(self, Op op):
        dims = self.get_input_dims(op, 0)
        data = np.zeros(shape=dims)
        val = array.array('f', data.flatten().tolist())
        cdef array.array arr = val
        self.p_graph.get_weight_value(op.guid, arr.data.as_floats)
        return val

    def get_split_lens(self, Op op):
        cdef int lens[128]
        cdef int numsplits = self.p_graph.get_split_lens(op.guid, lens)
        lenlist = list()
        for i in range(numsplits):
            lenlist.append(lens[i])
        return lenlist

    def get_output_dims(self, Op op, int idx):
        cdef int dims[8]
        cdef int ndims = self.p_graph.get_output_dims(op.guid, dims, idx)
        dimlist = list()
        for i in range(ndims):
            dimlist.append(dims[i])
        return dimlist

    def get_num_outputs(self, Op op):
        return self.p_graph.get_num_outputs(op.guid)

    def get_operator_type(self, Op op):
        cdef OpType type = self.p_graph.get_operator_type(op.guid)
        if type in op_table:
            return op_table[type]
        else:
            assert False, 'Undefined type: {}'.format(type)
            return "Undefined"

    def get_operator_attr(self, Op op, attrname):
        cdef int kh, kw, sh, sw
        cdef PaddingMode pm
        if attrname == 'kernel_shape':
            kh = self.p_graph.get_operator_int_attr(op.guid, PM_KERNEL_H)
            kw = self.p_graph.get_operator_int_attr(op.guid, PM_KERNEL_W)
            return [kh, kw]
        elif attrname == 'strides':
            sh = self.p_graph.get_operator_int_attr(op.guid, PM_STRIDE_H)
            sw = self.p_graph.get_operator_int_attr(op.guid, PM_STRIDE_W)
            return [sh, sw]
        elif attrname == 'pads':
            pm = <PaddingMode>self.p_graph.get_operator_int_attr(op.guid, PM_PAD)
            if pm == PD_MODE_VALID:
                return [0, 0, 0, 0]
            assert pm == PD_MODE_SAME
            dims = self.get_input_dims(op, 0)
            assert len(dims) == 4, "input tensor must be 4 dim for pads attribute"
            kh = self.p_graph.get_operator_int_attr(op.guid, PM_KERNEL_H)
            kw = self.p_graph.get_operator_int_attr(op.guid, PM_KERNEL_W)
            sh = self.p_graph.get_operator_int_attr(op.guid, PM_STRIDE_H)
            sw = self.p_graph.get_operator_int_attr(op.guid, PM_STRIDE_W)
            inputH = dims[2]
            inputW = dims[3]
            if inputH % sh == 0:
                padH = max(kh - sh, 0)
            else:
                padH = max(kh - (inputH % sh), 0)
            if inputW % sw == 0:
                padW = max(kw - sw, 0)
            else:
                padW = max(kw - (inputW % sw), 0)
            return [padH // 2, padW // 2, padH - padH // 2, padW - padW // 2]
        elif attrname == 'group':
            return self.p_graph.get_operator_int_attr(op.guid, PM_GROUP)
        elif attrname == 'axis':
            return self.p_graph.get_operator_int_attr(op.guid, PM_AXIS)
        elif attrname == 'split':
            return self.get_split_lens(op)
        elif attrname == 'perm':
            perIdx = self.p_graph.get_operator_int_attr(op.guid, PM_PERM)
            dims = self.get_output_dims(op, 0)
            for i in range(len(dims)-1,-1,-1):
                dims[i] = perIdx % len(dims)
                perIdx = perIdx // len(dims)
            perm = tuple(dims)
            return perm
        elif attrname == 'axes':
            # FIXME
            return [0]
        else:
           assert False, 'Internal error: unknow attribute {}'.format(attrname)
