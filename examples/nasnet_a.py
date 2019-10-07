import taso as ts
import onnx

def squeeze(graph, out_channels, input):
    weight = graph.new_weight(dims=(out_channels, input.dim(1), 1, 1))
    return graph.conv2d(input=input, weight=weight,
                        strides=(1, 1), padding="SAME",
                        activation="RELU")

def fit(graph, current, input):
    if input.dim(2) == current.dim(2):
        return squeeze(graph, current.dim(1), input)
    else:
        weight = graph.new_weight(dims=(current.dim(1), input.dim(1), 3, 3))
        return graph.conv2d(input=input, weight=weight, strides=(2, 2), padding="SAME", activation="RELU")

def seperable_conv(graph, input, out_channels, kernels, strides, padding, activation = "NONE"):
    assert input.dim(1) % out_channels == 0, "input.dim(1)={}, out_channels={}".format(input.dim(1), out_channels)
    weight1 = graph.new_weight(dims=(out_channels, input.dim(1) // out_channels, kernels[0], kernels[1]))
    t = graph.conv2d(input=input, weight=weight1, strides=strides, padding=padding)
    weight2 = graph.new_weight(dims=(out_channels, t.dim(1), 1, 1))
    return graph.conv2d(input=t, weight=weight2, strides=(1, 1), padding="SAME", activation=activation)

def normal_cell(graph, prev, cur, out_channels):
    cur = squeeze(graph, out_channels, cur)
    prev = fit(graph, cur, prev)
    ts = list()
    ts.append(seperable_conv(graph, input=cur, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(cur)
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(seperable_conv(graph, input=cur, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(graph.avgpool2d(input=cur, kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(prev)
    ts.append(graph.avgpool2d(input=prev, kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(graph.avgpool2d(input=prev, kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    assert len(ts) == 10, "Expected 10 tensors, got {}".format(len(ts))
    outputs = list()
    for i in range(5):
        outputs.append(graph.add(ts[2*i], ts[2*i+1]))
    return graph.concat(1, outputs)

def reduction_cell(graph, prev, cur, out_channels):
    cur = squeeze(graph, out_channels, cur)
    prev = fit(graph, cur, prev)
    ts = list()
    outputs = list()
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(7,7), strides=(2,2), padding="SAME"))
    ts.append(seperable_conv(graph, input=cur, out_channels=out_channels,
              kernels=(5,5), strides=(2,2), padding="SAME"))
    outputs.append(graph.add(ts[0], ts[1]))
    ts.append(graph.maxpool2d(input=cur, kernels=(3,3), strides=(2,2), padding="SAME"))
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(7,7), strides=(2,2), padding="SAME"))
    outputs.append(graph.add(ts[2], ts[3]))
    ts.append(graph.avgpool2d(input=cur, kernels=(3,3), strides=(2,2), padding="SAME"))
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(5,5), strides=(2,2), padding="SAME"))
    outputs.append(graph.add(ts[4], ts[5]))
    ts.append(graph.maxpool2d(input=cur, kernels=(3,3), strides=(2,2), padding="SAME"))
    ts.append(seperable_conv(graph, input=outputs[0], out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    outputs.append(graph.add(ts[6], ts[7]))
    ts.append(graph.avgpool2d(input=outputs[0], kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(outputs[1])
    outputs.append(graph.add(ts[8], ts[9]))
    return graph.concat(1, outputs)

graph = ts.new_graph()
input = graph.new_input(dims=(1,3,224,224))
weight = graph.new_weight(dims=(64,3,7,7))
input = graph.conv2d(input=input, weight=weight, strides=(2,2),
                 padding="SAME", activation="RELU")
input = graph.maxpool2d(input=input, kernels=(3,3), strides=(2,2), padding="SAME")

out_channels = 128
for i in range(3):
    prev = input
    cur = input
    for j in range(5):
        t = normal_cell(graph, prev, cur, out_channels)
        prev = cur
        cur = t
    out_channels *= 2
    input = reduction_cell(graph, prev, cur, out_channels)
new_graph = ts.optimize(graph, alpha=1.0, budget=100)
onnx_model = ts.export_onnx(new_graph)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, "nasneta_taso.onnx")
