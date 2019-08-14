import xflow as xf

def resnext_block(graph, input, strides, out_channels, groups):
    w1 = graph.new_weight(dims=(out_channels,input.dim(1),1,1))
    t = graph.conv2d(input=input, weight=w1,
                     strides=(1,1), padding="SAME",
                     activation="RELU")
    w2 = graph.new_weight(dims=(out_channels,t.dim(1)//groups,3,3))
    t = graph.conv2d(input=t, weight=w2,
                     strides=strides, padding="SAME",
                     activation="RELU")
    w3 = graph.new_weight(dims=(2*out_channels,t.dim(1),1,1))
    t = graph.conv2d(input=t, weight=w3,
                     strides=(1,1), padding="SAME")
    if (strides[0]>1) or (input.dim(1) != out_channels*2):
        w4 = graph.new_weight(dims=(out_channels*2,input.dim(1),1,1))
        input=graph.conv2d(input=input, weight=w4,
                           strides=strides, padding="SAME",
                           activation="RELU")
    return graph.relu(graph.add(input, t))

graph = xf.new_graph()
input = graph.new_input(dims=(1,64,56,56))
t = input
for i in range(3):
    t = resnext_block(graph, t, (1,1), 128, 64)
strides = (2,2)
for i in range(4):
    t = resnext_block(graph, t, strides, 256, 64)
    strides = (1,1)
strides = (2,2)
for i in range(6):
    t = resnext_block(graph, t, strides, 512, 64)
    strides = (1,1)
strides = (2,2)
for i in range(3):
    t = resnext_block(graph, t, strides, 1024, 64)
    strides = (1,1)

new_graph = xf.optimize(graph, alpha=1.0, budget=100)
#onnx_model = xf.export_onnx(new_graph)
#onnx.checker.check_model(onnx_model)
#onnx.save(onnx_model, "resnext50_xflow.onnx")
