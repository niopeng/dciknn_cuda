# DCI CUDA

This is the CUDA GPU implementation + Python interface (using PyTorch) of Dynamic Continuous Indexing (DCI) . The paper can be found [here](https://arxiv.org/abs/1512.00442).

## Prerequisites
* NVCC version >= 9.2 (Note: this should match the CUDA version that PyTorch is built with)
* PyTorch >= 1.4.0

## Setup

The library can be compiled using Python distutils.

**Note:** If your Python interpreter is named differently, e.g.: "python3", you will need to replace all occurrences of "python" with "python3" in the commands below.

If your Python installation is local (e.g. part of Anaconda), run the following command from the root directory of the code base to compile and install as a Python package:
```bash
python setup.py install
```

Otherwise, if you have sudo access, run the following command instead:
```bash
sudo python setup.py install
```

If you do not have sudo access, run the following command instead:
```bash
python setup.py install --user
```


## Getting Started

An example code using the PyTorch interface is provided. In the root directory of the code base, execute the following command:

```bash
python example.py
```


## Directory Layout
* `src`, all of the `*.cpp`, `.cu` files
* `include`, the header files
* `dciknn`, the Python interface

## Important Files
* `src/dci_cuda.cpp`: defines the PyTorch extension functions
* `src/util_kernel.cu`: matrix multiplication and random distribution generation functions
* `src/dci_cuda_kernel.cu`: main components of prioritized DCI
* `dciknn/core.py`: defines Python interface

## Reference

Please cite the following paper if you found this library useful in your research:

### [Fast _k_-Nearest Neighbour Search via Dynamic Continuous Indexing](https://arxiv.org/abs/1512.00442)
[Ke Li](https://people.eecs.berkeley.edu/~ke.li/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)\
*International Conference on Machine Learning (ICML)*, 2016

```
@inproceedings{li2016fast,
  title={Fast k-nearest neighbour search via {Dynamic Continuous Indexing}},
  author={Li, Ke and Malik, Jitendra},
  booktitle={International Conference on Machine Learning},
  pages={671--679},
  year={2016}
}
```
