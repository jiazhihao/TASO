import xflow
import onnx

# 1. evaluate the performance by just considering substitution optimizations
print("Measuring the performance of graph substitution optimizations (average of 1000 runs)")
graph = xflow.load('bert_graphs/bert_subst.onnx')
print("XFlow: end-to-end inference time = {}ms".format(graph.run_time()))
print()

#2. evaluate the performance by just performing data layout optimizations
print("Measuring the performance of data layout optimizations")
graph = xflow.load('bert_graphs/bert_layout.onnx')
print("XFlow: end-to-end inference time = {}ms".format(graph.run_time()))
print()

#3. evaluate the performance by sequential optimizations
print("Measuring the performance of sequential optimizations")
graph = xflow.load('bert_graphs/bert_sequential.onnx')
print("XFlow: end-to-end inference time = {}ms".format(graph.run_time()))
print()

#4. evaluate the performance by joint optimizations
print("Measuring the performance of joint optimizations")
graph = xflow.load('bert_graphs/bert_xflow.onnx')
print("XFlow: end-to-end inference time = {}ms".format(graph.run_time()))
print()

