'''
Code for Fast k-Nearest Neighbour Search via Prioritized DCI

This code implements the method described in the Prioritized DCI paper,
which can be found at https://arxiv.org/abs/1703.00440

This file is a part of the Dynamic Continuous Indexing reference
implementation.


This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Copyright (C) 2020    Ke Li, Shichong Peng, Mehran Aghabozorgi
'''
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths
import os
import sys

if sys.version_info[0] < 3:
    with open('README.md') as f:
        long_description = f.read()
else:
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='dciknn_cuda',
    packages=['dciknn_cuda'],
    version='0.1.11',    
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='DCI CUDA for fast K nearest neighbour finding',
    url='https://github.com/niopeng/dciknn_cuda',
    author='Ke Li, Shichong Peng, Mehran Aghabozorgi',
    author_email='keli@sfu.ca',
    license='Mozilla Public License Version 2.0',
    install_requires=['torch>=1.4.0'],
    include_dirs=include_paths(),
    language='c++',
    soruces=['./src/dci_cuda.cpp',
            './src/dci_cuda_kernel.cu',
            './src/util_kernel.cu',],
    ext_modules=[
        CUDAExtension('_dci_cuda', [
            './src/dci_cuda.cpp',
            './src/dci_cuda_kernel.cu',
            './src/util_kernel.cu',
        ], include_dirs=[
            os.path.abspath(os.path.join(os.path.dirname(__file__), 'include')),
        ], extra_compile_args={
                'cxx': ['-g'],
                'gcc': ['-g', '-o'],
                'nvcc': ['-g', '-G'] 
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
