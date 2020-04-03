# TASO: A Tensor Algebra SuperOptimizer for Deep Learning

TASO optimizes the computation graphs of DNN models using automatically generated and verified graph transformations.
For an arbitrary DNN model, TASO uses the auto-generated graph transformations to build a large search space of potential computation graphs that are equivalent to the original DNN model.
TASO employs a cost-based search algorithm to explore the space, and automatically discovers highly optimized computation graphs.
TASO outperforms the graph optimizers in existing deep learning frameworks by [up to 3x](http://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf).
<div align="center">
  <img src="https://github.com/jiazhihao/TASO/blob/master/figures/inference.png">
  End-to-end inference performance comparison on a NVIDIA V100 GPU.
</div>

## Install TASO

See [instructions](INSTALL.md) to install TASO from source.
We also provide prebuilt [docker images](https://github.com/jiazhihao/TASO/blob/master/INSTALL.md) with all dependencies pre-installed.

## Use TASO

TASO can directly optimize any pre-trained DNN models in [ONNX](https://onnx.ai), [TensorFlow](https://www.tensorflow.org/guide/saved_model), and [PyTorch](https://pytorch.org/docs/stable/onnx.html) graph formats.
TASO also provides a Python interface for optimizing arbitrary DNN architectures.
TASO supports exporting the optimized computation graphs to ONNX, which can be directly used as inputs by most existing deep learning frameworks.

### Optimize ONNX Models

TASO can directly optimize pre-trained ONNX models, and this can be done in just a few lines of Python code.
The following code snippet shows how to load a pre-trained DNN model from ONNX, optimize the model, and save the optimized model into a ONNX file.
```python
import taso
import onnx

old_model = taso.load_onnx("/path/to/load/onnx/model")
taso_graph = taso.optimize(old_model)
new_model = taso.export_onnx(taso_graph)
onnx.save(new_model, "/path/to/save/new/onnx/model")
```
The optimized model has the same accuracy as the original and can be directly used by existing deep learning frameworks.
Some original and TASO-optimized ONNX files are available in the `onnx` folder.
<!-- The following figure shows the end-to-end inference performance comparison on a NVIDIA V100 GPU. -->

### Optimize TensorFlow Models

TASO can optimize TensorFlow models by converting the model to ONNX using [tf2onnx](https://github.com/onnx/tensorflow-onnx).

* First, install `tf2onnx` from PyPi as follows or [from source](https://github.com/onnx/tensorflow-onnx).
```
pip install -U tf2onnx
```

* Second, convert a TensorFlow model to ONNX using `tf2onnx`.
```
python -m tf2onnx.convert \
       --saved-model /path/to/tensorflow/saved/model \
       --output /path/to/onnx/model/file
```

* Third, use TASO to optimize the model in ONNX by following the [above instructions](https://github.com/jiazhihao/TASO#optimize-onnx-models).

### Optimize PyTorch Models

PyTorch has built-in support for ONNX as a part of the [torch.onnx](https://pytorch.org/docs/master/onnx.html) package.
TASO can directly optimize PyTorch models in the ONNX format.

### Optimize Arbitrary DNN Models using the Python Interface

TASO can also optimize arbitrary DNN architectures using the TASO Python interface. 
The following code snippet builds the left-most DNN graph depicted in the figure. TASO automatically performs a series of non-trivial transformations, and eventually discovers the right-most DNN graph, which is 1.3x faster on a V100 GPU. More DNN examples are available in the `examples` folder.

<div align="center">
  <img src="https://github.com/jiazhihao/TASO/blob/master/figures/graph_subst.png">
</div>

```python
import taso
import onnx

#Build DNN model
graph = taso.new_graph()
input = graph.new_input(dims=(1,128,56,56))
w1 = graph.new_weight(dims=(128,128,3,3))
w2 = graph.new_weight(dims=(128,128,1,1))
w3 = graph.new_weight(dims=(128,128,3,3))
left = graph.conv2d(input=input, weight=w1, strides=(1,1), padding="SAME", activation="RELU")
left = graph.conv2d(input=left, weight=w3, strides=(1,1), padding="SAME")
right = graph.conv2d(input=input, weight=w2, strides=(1,1), padding="SAME", activation="RELU")
output = graph.add(left, right)
output = graph.relu(output)

#Optimize DNN model
new_graph = taso.optimize(graph)
onnx_model = taso.export_onnx(new_graph)
onnx.save(onnx_model, "/path/to/save/new/onnx/model")
```

## Publication
* Zhihao Jia, Oded Padon, James Thomas, Todd Warszawski, Matei Zaharia, and Alex Aiken. [TASO: Optimizing Deep Learning Computation with Automated Generation of Graph Substitutions](https://cs.stanford.edu/~zhihao/papers/sosp19.pdf). In Proceedings of the Symposium on Operating Systems Principles (SOSP), Ontario, Canada, October 2019.

* Zhihao Jia, James Thomas, Todd Warszawski, Mingyu Gao, Matei Zaharia, and Alex Aiken. [Optimizing DNN Computation with Relaxed Graph Substitutions](https://theory.stanford.edu/~aiken/publications/papers/sysml19b.pdf). In Proceedings of the Conference on Systems and Machine Learning (SysML), Palo Alto, CA, April 2019.

