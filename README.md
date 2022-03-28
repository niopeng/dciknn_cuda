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


## Experimental PyPI install
Simply run:
```bash
pip install -i https://test.pypi.org/simple/ dciknn-cuda==0.1.10
```
If you don't have internet access (e.g., in a requested job in clusters), you can run the following before requesting the job:
```bash
pip download -i https://test.pypi.org/simple/ dciknn-cuda==0.1.10
```
Then run the following in the requested job:
```
pip install dciknn_cuda-0.1.10.tar.gz
```


## Getting Started

An example code using the PyTorch interface is provided. In the root directory of the code base, execute the following command:

```bash
python example.py
```

### Multi-GPU example
The multi-GPU version of DCI exposes the same APIs to be used. The following is a simple example for using four GPUs for computing nearest neighbours:
```python
# Multi-GPU version of DCI
dci_db = MDCI(dim, num_comp_indices, num_simp_indices, block_size, thread_size, devices=[0, 1, 2, 3])  # We specify GPUs to be used by the DCI instance with `devices`. Set to list(range(torch.cuda.device_count())) to use all available GPUs

        dci_db.add(data)  # We add the pool of data
        indices, dists = dci_db.query(query, num_neighbours, num_outer_iterations)  # We run our desired query
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
