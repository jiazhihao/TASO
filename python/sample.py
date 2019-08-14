import xflow
import onnx

graph = xflow.new_graph()
input = graph.new_input(dims=(1,256,28,28))
input = graph.maxpool2d(input=input, kernels=(1,1), strides=(1,1), padding="SAME")
weight1 = graph.new_weight(dims=(256,8,3,3))
#weight2 = graph.new_weight(dims=(256,16,3,3))
#weight3 = graph.new_weight(dims=(256,32,3,3))
#weight4 = graph.new_weight(dims=(256,64,3,3))
#weight5 = graph.new_weight(dims=(256,128,3,3))
#weight6 = graph.new_weight(dims=(256,256,3,3))
t1 = graph.conv2d(input=input,weight=weight1,strides=(1,1), padding="SAME", activation="RELU")
#t2 = graph.conv2d(input=input,weight=weight2,strides=(1,1), padding="SAME", activation="RELU")
#t3 = graph.conv2d(input=input,weight=weight3,strides=(1,1), padding="SAME", activation="RELU")
#t4 = graph.conv2d(input=input,weight=weight4,strides=(1,1), padding="SAME", activation="RELU")
#t5 = graph.conv2d(input=input,weight=weight5,strides=(1,1), padding="SAME", activation="RELU")
#t6 = graph.conv2d(input=input,weight=weight6,strides=(1,1), padding="SAME", activation="RELU")

new_graph = xflow.optimize(graph, alpha=1.0, budget=100)
onnx_model = xflow.export_onnx(new_graph)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, "/home/ubuntu/ONNXModel/inception_v2/model_xflow.onnx")
