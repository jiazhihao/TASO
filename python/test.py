import taso
import onnx

graph = taso.load("/home/ubuntu/taso/onnx/squeezenet1.1.onnx") 
#graph = xflow.load("/home/ubuntu/resnext-101.onnx") 
#graph = xflow.load("/home/ubuntu/ONNXModel/inception_v2/model.onnx")
new_graph = taso.optimize(graph, alpha = 1.0, budget = 100)
onnx_model = taso.export_onnx(new_graph)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, "/home/ubuntu/taso/onnx/squeezenet/model_taso.onnx")
