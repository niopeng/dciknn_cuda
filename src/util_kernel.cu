/*
 * Code for Fast k-Nearest Neighbour Search via Prioritized DCI
 *
 * This code implements the method described in the Prioritized DCI paper,
 * which can be found at https://arxiv.org/abs/1703.00440
 *
 * This file is a part of the Dynamic Continuous Indexing reference
 * implementation.
 *
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * Copyright (C) 2020    Ke Li, Shichong Peng
 */

#include "util.h"
// Utilities and system includes
#include <assert.h>
#include <malloc.h>
#include <math.h>

// generate the random seed
#include <inttypes.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA random
#include <curand.h>
#include <curand_kernel.h>
#include <unistd.h>


#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

// uses device pointers, save on malloc ops
void matmul_device(const cublasOperation_t op_A, const cublasOperation_t op_B,
    const int M, const int N, const int K, const float* const A, const float* const B, float* const C, int &devID) {
    // initialize the CUDA variables
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, devID);
    int block_size = 32;  // size 16 has also been used. Think 32 is faster

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(N / threads.x, M / threads.y);

    // CUBLAS version 2.0
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasHandle_t handle;

    cublasCreate(&handle);

    int lda, ldb;
    if(op_A == CUBLAS_OP_N) {
        lda = K;
    } else {
        lda = M;
    }
    if(op_B == CUBLAS_OP_N) {
        ldb = N;
    } else {
        ldb = K;
    }

    cublasSgemm(handle, op_B, op_A, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);

    // Destroy the handle
    cublasDestroy(handle);
}

__global__ void init_curand_state(unsigned int seed, curandState_t* states) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}


// gauss random variables in parallel
__global__ void
gauss_parallel_rng(curandState_t* states, float* vec, const int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // Note assumes num_blocks = num_threads
    int chunk_size = (n + blockDim.x * blockDim.x - 1) / (blockDim.x * blockDim.x);
    int index;
    for(int j = 0; j < chunk_size; ++j) {
        index = i*chunk_size+j;
        if(index < n) {
            vec[i*chunk_size+j] = curand_normal(&states[i]);
        }
    }
}

// uniform distribution in [-1, 1] in parallel
__global__ void
uniform_parallel_rng(curandState_t* states, float *vec, const int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // Note assumes num_blocks = num_threads
    int chunk_size = (n + blockDim.x * blockDim.x - 1) / (blockDim.x * blockDim.x);
    int index;
    for(int j = 0; j < chunk_size; ++j) {
        index = i*chunk_size+j;
        if(index < n) {
            vec[i*chunk_size+j] = (curand_uniform(&states[i]) * 2.0) - 1.0;
        }
    }
}

// helper functon, assumes vec is device pointer
void rng_parallel_device(float* const vec, const int n, const int rng_type) {
    int num_blocks = 64;  //  for now using num_blocks blocks, num_blocks threads per block

    // curand initialization
    curandState_t* states;
    long long seed = 0;
    for(int i = 0; i < 4; ++i) {
        seed = (seed << 32) | rand();
    }
    cudaMalloc((void**) &states, num_blocks * num_blocks * sizeof(curandState_t));
    init_curand_state<<<num_blocks, num_blocks>>>(seed, states);

    // generate random numbers
    if(rng_type == GAUSS_RAND) {
        gauss_parallel_rng<<<num_blocks, num_blocks>>>(states, vec, n);
    } else {
        uniform_parallel_rng<<<num_blocks, num_blocks>>>(states, vec, n);
    }
}
