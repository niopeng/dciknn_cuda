# DCI CUDA

This is the CUDA GPU implementation + Python interface (using PyTorch) of prioritized DCI. The paper can be found at https://arxiv.org/abs/1703.00440.

### Installation Instructions
* NVCC version >= 8.0 (Note: this should match the CUDA version that PyTorch built with)
* PyTorch >= 1.4.0
* run `python setup.py install`

### Directory Layout
* `src`, all of the `*.cpp`, `.cu` files
* `include`, the header files
* `dciknn`, the Python interface

### Important Files
* `src/dci_cuda.cpp`: defines the PyTorch extension functions
* `src/util_kernel.cu`: matrix multiplication and random distribution generation functions
* `src/dci_cuda_kernel.cu`: main components of prioritized DCI
* `dciknn/core.py`: defines Python interface
