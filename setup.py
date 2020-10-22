'''
Code for Fast k-Nearest Neighbour Search via Prioritized DCI

This code implements the method described in the Prioritized DCI paper,
which can be found at https://arxiv.org/abs/1703.00440

This file is a part of the Dynamic Continuous Indexing reference
implementation.


This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Copyright (C) 2020    Ke Li, Shichong Peng
'''
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='dciknn_cuda',
    ext_modules=[
        CUDAExtension('_dci_cuda', [
            './src/dci_cuda.cpp',
            './src/dci_cuda_kernel.cu',
            './src/util_kernel.cu',
        ], include_dirs=[
            os.path.abspath(os.path.join(os.path.dirname(__file__), 'include')),
        ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
