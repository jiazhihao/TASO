# TASO Installation

TASO can be built from source code using the following instructions.
We also provide prebuilt TASO docker images with all dependencies preinstalled.

## Install from Source

### Prerequisties

* CMAKE 3.2 or higher
* ProtocolBuffer 3.6.1 or higher
* Cython 0.28 or higher
* ONNX 1.5 or higher
* CUDA 9.0 or higher and CUDNN 7.0 or higher

### Build TASO Runtime

* To get started, clone the TASO source code from github.
```
git clone --recursive https://www.github.com/jiazhihao/taso
cd taso
```

* Build the TASO runtime library. The configuration of the TASO runtime can be modified by `config.cmake`. The default configuration builds the CUDA backend and automatically finds the CUDA libraries (e.g., cuDNN, cuBLAS). You can manually choose a CUDA path by changing `set(USE_CUDA ON)` to `set(USE_CUDA /path/to/cuda/library`). MKL support is coming soon.
```
mkdir build; cd build; cmake ..
sudo make install -j 4
```

* Install the TASO python package.
```
cd python
python setup.py install
```

## Docker Images

We require [docker](https://docs.docker.com/engine/installation/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/) to run the TASO [docker images](https://hub.docker.com/u/tasoml).

* First, clone the TASO gitpub repository to obtain the necessary scripts
```
git clone --recursive https://www.github.com/jiazhihao/taso
```

* Second, we can use the following command to run a TASO docker image for CUDA 10.0.
```
/path/to/taso/docker/run_docker.sh tasoml/cuda100
```

* You are ready to use TASO now. Try some of our example DNN architectures.
```
python /path/to/taso/examples/resnext10.py
```
