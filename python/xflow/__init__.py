from .core import *
import onnx
from onnx import helper, TensorProto, numpy_helper

def _check_output(xf_output, onnx_output):
    # TODO: check output match
    return True

def _parse_attribute(attributes):
    atts = dict()
    for att in attributes:
        if att.type == onnx.AttributeProto.INT:
            atts[att.name] = att.i
        elif att.type == onnx.AttributeProto.INTS:
            atts[att.name] = att.ints
        elif att.type == onnx.AttributeProto.FLOAT:
            atts[att.name] = att.f
        elif att.type == onnx.AttributeProto.STRING:
            atts[att.name] = att.s
        else:
            assert False, "Unsupported Attribute Type: {}".format(att.type)
    return atts

def _get_inputs(op, tensors):
    inputs = list()
    for i in op.input:
        assert i in tensors, "Input tensor not found"
        inputs.append(tensors[i])
    return inputs

def _add(op, graph, tensors, initializer):
    inputs = _get_inputs(op, tensors)
    outputs = graph.add(inputs[0], inputs[1]);
    return outputs;

def _batchnorm(op, graph, tensors, initializer):
    inputs = _get_inputs(op, tensors)
    attrs = _parse_attribute(op.attribute)
    outputs = graph.batchnorm(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
    return outputs

def _concat(op, graph, tensors, initializer):
    inputs = _get_inputs(op, tensors)
    attrs = _parse_attribute(op.attribute)
    axis = attrs["axis"]
    outputs = graph.concat(axis, inputs)
    return outputs

def _conv2d(op, graph, tensors, initializer):
    inputs = _get_inputs(op, tensors)
    attrs = _parse_attribute(op.attribute)
    if "group" not in attrs:
        group = 1 # default 1
    else:
        group = attrs["group"]
    # Note that we always think conv1x1 has SAME padding
    # This will allow fusing enlarged convs
    if sum(attrs["pads"]) == 0 and sum(attrs['kernel_shape']) > 2:
        pads = "VALID"
    else:
        pads = "SAME"
    strides = attrs["strides"]
    outputs = graph.conv2d(input=inputs[0], weight=inputs[1], strides=strides, padding=pads)
    return outputs

def _dropout(op, graph, tensors, initializer):
    inputs = _get_inputs(op, tensors)
    assert len(inputs) == 1, "Dropout requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    rate = attrs["ratio"]
    outputs = graph.dropout(inputs[0], rate)
    return outputs

def _matmul(op, graph, tensors, initializer):
    inputs = _get_inputs(op, tensors)
    assert len(inputs) == 2, "Matmul requires exactly two inputs"
    outputs = graph.matmul(inputs[0], inputs[1])
    return outputs

def _pad(op, graph, tensors, initializer):
    inputs = _get_inputs(op, tensors)
    attrs = _parse_attribute(op.attribute)
    # Currently treat pad as a no op
    assert sum(attrs['pads']) == 0
    return inputs

def _maxpool2d(op, graph, tensors, initializer):
    assert len(op.input) == 1, "MaxPool2D requires exactly one input"
    assert op.input[0] in tensors
    attrs = _parse_attribute(op.attribute)
    kernels = attrs["kernel_shape"]
    strides = attrs["strides"]
    if sum(attrs["pads"]) == 0:
        pads = "VALID"
    else:
        pads = "SAME"
    outputs = graph.maxpool2d(input=tensors[op.input[0]], kernels=kernels, strides=strides, padding=pads)
    return outputs

def _avgpool2d(op, graph, tensors, initializer):
    assert len(op.input) == 1, "MaxPool2D requires exactly one input"
    assert op.input[0] in tensors
    attrs = _parse_attribute(op.attribute)
    kernels = attrs["kernel_shape"]
    strides = attrs["strides"]
    if sum(attrs["pads"]) == 0:
        pads = "VALID"
    else:
        pads = "SAME"
    outputs = graph.avgpool2d(input=tensors[op.input[0]], kernels=kernels, strides=strides, padding=pads)
    return outputs

def _reshape(op, graph, tensors, initializer):
    inputs = _get_inputs(op, tensors)
    assert len(inputs) == 2
    for data in initializer:
        if data.name == op.input[1]:
            shape = list()
            for dim in data.int64_data:
                shape.append(dim)
    outputs = graph.reshape(inputs[0], tuple(shape))
    return outputs

def _relu(op, graph, tensors, initializer):
    assert len(op.input) == 1, "Relu requires exactly one input"
    assert op.input[0] in tensors
    attrs = _parse_attribute(op.attribute)
    outputs = graph.relu(tensors[op.input[0]])
    return outputs

def _split(op, graph, tensors, initializer):
    assert len(op.input) == 1, "Split requires exactly one input"
    assert op.input[0] in tensors
    attrs = _parse_attribute(op.attribute)
    axis = attrs["axis"]
    split_ints = attrs["split"]
    split_list = list()
    for i in split_ints:
        split_list.append(i)
    outputs = graph.split(tensors[op.input[0]], axis, split_list)
    return outputs

def _transpose(op, graph, tensors, initializer):
    assert len(op.input) == 1, "Transpose requires exactly one input"
    assert op.input[0] in tensors
    attrs = _parse_attribute(op.attribute)
    perm_ints = attrs["perm"]
    perm = list()
    for i in perm_ints:
        perm.append(i)
    outputs = graph.transpose(tensors[op.input[0]], tuple(perm), shuffle=True)
    return outputs

# Add all supported operators
xf_operators = dict()
xf_operators['Add'] = _add
xf_operators['BatchNormalization'] = _batchnorm
xf_operators['Concat'] = _concat
xf_operators['Conv'] = _conv2d
xf_operators['Dropout'] = _dropout
xf_operators['Pad'] = _pad
xf_operators['Reshape'] = _reshape
xf_operators['Relu'] = _relu
xf_operators['Matmul'] = _matmul
xf_operators['MaxPool'] = _maxpool2d
xf_operators['AveragePool'] = _avgpool2d
xf_operators['Split'] = _split
xf_operators['Transpose'] = _transpose

def new_graph(print_measurements = False):
    graph = core.PyGraph()
    if print_measurements:
        graph.print_measurements()
    return graph

def load(filename):
    '''
    Load a onnx file and return a Graph

    @params
    filename is a string containing a file name
    
    @return
    Loaded in-memory Graph
    '''
    graph = core.PyGraph()
    model = onnx.load(filename)
    tensors = dict()
    for t in model.graph.input:
        dims = list()
        for d in t.type.tensor_type.shape.dim:
            dims.append(d.dim_value)
        if "data" in t.name:
            tensors[t.name] = graph.new_input(dims=tuple(dims))
        else:
            weight_data = None
            for weight in model.graph.initializer:
                if (weight.name == t.name):
                    weight_data = numpy_helper.to_array(weight)
            assert(weight_data is not None)
            tensors[t.name] = graph.new_weight(dims=tuple(dims), data = weight_data)

    for op in model.graph.node:
        if op.op_type in xf_operators:
            outputs = xf_operators[op.op_type](op, graph, tensors, model.graph.initializer)
            if not isinstance(outputs, list):
                outputs = [outputs]
            assert len(outputs) == len(op.output), "Number of output tensors mismatch"
            for i in range(len(outputs)):
                assert _check_output(outputs[i], op.output[i])
                tensors[op.output[i]] = outputs[i]
        else:
            assert False, "Unsupported ONNX operator: {}".format(op.op_type)
    return graph

input_weight_names = dict()
input_weight_names['Conv'] = ['input', 'weight', 'bias']
input_weight_names['Matmul'] = ['input', 'weight']
input_weight_names['Reshpe'] = ['input', 'shape']

operator_attrs = dict()
operator_attrs['Add'] = []
operator_attrs['AveragePool'] = ['kernel_shape', 'pads', 'strides']
operator_attrs['Concat'] = ['axis']
operator_attrs['Conv'] = ['group', 'kernel_shape', 'pads', 'strides']
operator_attrs['Dropout'] = []
operator_attrs['Matmul'] = []
operator_attrs['MaxPool'] = ['kernel_shape', 'pads', 'strides']
operator_attrs['Split'] = ['axis', 'split']
operator_attrs['Relu'] = []
operator_attrs['Reshape'] = []
operator_attrs['Transpose'] = ['perm']

def _input_tensor_name(graph, inedge, op):
    intype = graph.get_operator_type(inedge['srcOp'])
    if intype == "Input":
        return "data"
    elif intype == "Weight":
        mytype = graph.get_operator_type(op)
        return "{}{}_{}".format(mytype, op['guid'], input_weight_names[mytype][inedge['dstIdx']])
    else:
        return _output_tensor_name(graph, inedge['srcOp'], inedge['srcIdx'])

def _output_tensor_name(graph, op, idx):
    type = graph.get_operator_type(op)
    return "{}{}_fwd{}".format(type, op['guid'], idx)

def _add_node_attribute(graph, node, op, optype):
    for key in operator_attrs[optype]:
        val = graph.get_operator_attr(op, key)
        attr = helper.make_attribute(key, val)
        node.attribute.append(attr)

def export_onnx(graph):
    '''
    Export a XFlow graph to an ONNX graph
    
    @params
    graph is a XFlow graph

    @return
    A in-memory ONNX graph
    '''
    opList = graph.get_operator_list()
    graph_nodes = list()
    graph_inputs = list()
    graph_initializers = list()
    graph_outputs = list()
    output_guids = dict()
    for op in opList:
        mytype = graph.get_operator_type(op)
        inedges = graph.get_input_edges(op)
        print("op.guid={} mytype={} inedges={}".format(op['guid'], mytype, len(inedges)))
        inputs = list()
        for e in inedges:
            intype = graph.get_operator_type(e['srcOp'])
            inputs.append(_input_tensor_name(graph, e, op))
            output_guids.pop((e['srcOp']['guid'], e['srcIdx']), None)
            if intype == 'Input' or intype == 'Weight':
                graph_inputs.append(helper.make_tensor_value_info(_input_tensor_name(graph, e, op),
                                    TensorProto.FLOAT, graph.get_input_dims(op, e['dstIdx'])))
            if intype == 'Weight':
                graph_initializers.append(helper.make_tensor(_input_tensor_name(graph, e, op),
                                          TensorProto.FLOAT, graph.get_input_dims(op, e['dstIdx']),
                                          graph.get_weight_value(e['srcOp'])))

        # add a second input for Reshape
        if mytype == 'Reshape':
            inputs.append('Reshape_attr{}'.format(op['guid']))
            shape = graph.get_output_dims(op, 0)
            graph_inputs.append(helper.make_tensor_value_info('Reshape_attr{}'.format(op['guid']), TensorProto.INT64, [len(shape)]))
            graph_initializers.append(helper.make_tensor('Reshape_attr{}'.format(op['guid']), TensorProto.INT64, [len(shape)], shape))
        outputs = list()
        for i in range(graph.get_num_outputs(op)):
            outputs.append(_output_tensor_name(graph, op, i))
            output_guids[(op['guid'], i)] = op
        node = helper.make_node(mytype, inputs, outputs, '{}{}'.format(mytype, op['guid']))
        _add_node_attribute(graph, node, op, mytype)
        graph_nodes.append(node)
    for guid, idx in output_guids:
        op = output_guids[(guid, idx)]
        graph_outputs.append(helper.make_tensor_value_info(_output_tensor_name(graph, op, idx),
                             TensorProto.FLOAT, graph.get_output_dims(op, idx)))
    onnx_graph = helper.make_graph(graph_nodes, 'main', graph_inputs, graph_outputs, graph_initializers)
    onnx_model = helper.make_model(onnx_graph, producer_name='XFlow Optimized Model')
    return onnx_model

def optimize(graph, alpha = 1.0, budget = 1000):
    return graph.optimize(alpha, budget)
