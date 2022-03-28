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
 * Copyright (C) 2020    Ke Li, Shichong Peng, Mehran Aghabozorgi
 */

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA random
#include <curand.h>
#include <curand_kernel.h>
#include <unistd.h>

#ifndef UTIL_H
#define UTIL_H

#define GAUSS_RAND 0
#define UNIFORM_RAND 1

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
        unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

// put in device pointers. Saves on memcpy operations
void matmul_device(const cublasOperation_t op_A, const cublasOperation_t op_B,
    const int M, const int N, const int K, const float* const A, const float* const B, float* const C, int &devID);

// put in device pointers. Saves on memcpy operations
void rng_parallel_device(float* const vec, const int n, const int rng_type);

__global__ void init_curand_state(unsigned int seed, curandState_t* states);

__global__ void gauss_parallel_rng(curandState_t* states, float *vec, const int n);

__global__ void uniform_parallel_rng(curandState_t* states, float *vec, const int n);

#endif // UTIL_H
