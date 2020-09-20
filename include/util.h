/*
 * Code for Fast k-Nearest Neighbour Search via Prioritized DCI
 *
 * This code implements the method described by Li et al., which can be found at https://arxiv.org/abs/1703.00440
 * This code also builds off of code written by Ke Li.
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
