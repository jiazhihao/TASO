# TASO: A Tensor Algebra SuperOptimizer for Deep Learning

TASO is a Tensor Algebra SuperOptimizer that automatically optimizes deep neural network architectures with preserving the original network accuracy.

## Installation

### Install from Source

* To get started, clone the TASO source code from github.
```
git clone https://www.github.com/jiazhihao/taso
cd taso
```

* Build the TASO runtime library. The configuration of the TASO runtime can be modified by `config.cmake`. The default configuration only builds the CUDA backend, and you can change `set(USE_MKL OFF)` to `set(USE_MKL ON` to enable the MKL CPU backend.
```
mkdir build; cd build; cmake ..
sudo make install -j 4
```

* Install the TASO python package.
```
cd python
python setup.py install
```

## Use TASO to Optimize DNN Computation

### Optimize Pre-trained ONNX Graphs

TASO can be used to optimize pre-trained DNN models in the [ONNX](https://onnx.ai/) format. The following code snippet shows how to load a pre-trained DNN model, optimize the model, and save the optimized model to a ONNX file. The optimized model can be directly used by existing deep learning frameworks to achieve optimized performance.

```python
import taso
import onnx

old_model = taso.load("/path/to/load/onnx/model")
taso_graph = taso.optimize(old_model)
new_model = taso.export_onnx(taso_graph)
onnx.save(new_model, "/path/to/save/new/onnx/model")
```

### Build DNN Architectures from Scratch

The following code snippet shows how to build the left-most DNN graph depicted in the figure. TASO automatically performs a series of non-trivial transformations, and eventually discovers the right-most DNN graph, which is 1.3x faster on a V100 GPU. More example DNN architectures are available in the `examples` subfolder.

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
left = graph.conv2d(input=input, weight=w3, strides=(1,1), padding="SAME")
right = graph.conv2d(input=input, weight=w2, strides(1,1), padding="SAME", activation="RELU")
output = graph.add(left, right)
output = graph.relu(output)

#Optimize DNN model
new_graph = taso.optimize(graph)
onnx_model = taso.export_onnx(new_graph)
onnx.save(onnx_model, "/path/to/save/new/onnx/model")
```

## Publication
* Zhihao Jia, Oded Padon, James Thomas, Todd Warszawski, Matei Zaharia, and Alex Aiken. [TASO: Optimizing Deep Learning Computation with Automated Generation of Graph Substitutions](http://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf). In Proceedings of the Symposium on Operating Systems Principles (SOSP), Ontario, Canada, October 2019.

* Zhihao Jia, James Thomas, Todd Warszawski, Mingyu Gao, Matei Zaharia, and Alex Aiken. [Optimizing DNN Computation with Relaxed Graph Substitutions](https://theory.stanford.edu/~aiken/publications/papers/sysml19b.pdf). In Proceedings of the Conference on Systems and Machine Learning (SysML), Palo Alto, CA, April 2019.

