# DCI CUDA

This is the CUDA GPU implementation + Python interface (using PyTorch) of prioritized DCI. The paper can be found at https://arxiv.org/abs/1703.00440.

### Prerequisites
* NVCC version >= 8.0 (Note: this should match the CUDA version that PyTorch is built with)
* PyTorch >= 1.4.0
* run `python setup.py install`

### Setup

The library can be compiled using Python distutils.

**Note:** If your Python interpreter is named differently, e.g.: "python3", you will need to replace all occurrences of "python" with "python3" in the commands below.

If your Python installation is local (e.g. part of Anaconda), run the following command:
```bash
python setup.py install
```

Otherwise, if you have sudo access, run the following command from the root directory of the code base to compile and install as a Python package:
```bash
sudo python setup.py install
```

If you do not have sudo access, run the following command instead:
```bash
python setup.py install --user
```

### Directory Layout
* `src`, all of the `*.cpp`, `.cu` files
* `include`, the header files
* `dciknn`, the Python interface

### Important Files
* `src/dci_cuda.cpp`: defines the PyTorch extension functions
* `src/util_kernel.cu`: matrix multiplication and random distribution generation functions
* `src/dci_cuda_kernel.cu`: main components of prioritized DCI
* `dciknn/core.py`: defines Python interface


