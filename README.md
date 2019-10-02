TASO: A Tensor Algebra SuperOptimizer for Deep Learning
===============================================

TASO is a Tensor Algebra SuperOptimizer that automatically optimizes deep neural network architectures with preserving the original network accuracy.

## Installation

### Install from Source

* To get started, clone the TASO source code from github.
```
git clone https://www.github.com/jiazhihao/taso
```

* Build the TASO runtime library. The configuration of the TASO runtime can be modified by `config.cmake`. The default configuration only builds the CUDA backend, and you can change `set(USE_MKL OFF)` to `set(USE_MKL ON` to enable the MKL CPU backend.
```
mkdir build; cd build; make -j 4
sudo make install
```

* Install the TASO python package.
```
cd python
python setup.py install
```
