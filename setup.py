from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='dci',
    ext_modules=[
        CUDAExtension('_dci_cuda', [
            './src/dci_cuda.cpp',
            './src/dci_cuda_kernel.cu',
            './src/util_kernel.cu',
        ], include_dirs=[
            os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'include')),
        ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
