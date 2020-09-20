#include "dci.h"
#include "util.h"
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <stdlib.h>

// generate the random seed
#include <inttypes.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

void small_scale(const int num_points) {
	const int dim = 1000;
	const int num_query_points = 1; // number of points to query for K-NN
	const int num_comp_indices = 2;
	const int num_simp_indices = 5;
	const int num_neighbours = 1; // k in k-NN
	const int num_queries = 1;
	int num_outer_iterations = 5000;
	int max_num_candidates = 10 * num_neighbours; // do this so num_neighbors < max_num_candidates
//	const int max_num_candidates = 100;
	int block_size = 2;
	int thread_size = 2;

	dci *py_dci_inst;
    cudaMallocManaged((void **) &py_dci_inst, sizeof(dci));

    dci_init(py_dci_inst, dim, num_comp_indices, num_simp_indices);

    float *data;

    cudaMallocManaged((void **) &data, sizeof(float) * 1000);

    dci_add(py_dci_inst, dim, num_points, data, block_size, thread_size);

    float *query;

    cudaMallocManaged((void **) &query, sizeof(float) * 1000);

    int*  final_outputs;
    float* final_distances;
    const int output_size = num_neighbours * num_queries;
    cudaMalloc((void **) &(final_outputs), sizeof(int) * output_size);
    cudaMalloc((void **) &(final_distances), sizeof(float) * output_size);

    dci_query_config query_config = {false, num_outer_iterations, max_num_candidates};

    // query using DCI
    dci_query(py_dci_inst, dim, num_queries, query, num_neighbours,
      query_config, final_outputs, final_distances, block_size, thread_size);

	printf("Reached end.\n");
	printf("done with %d points\n", num_points);

	dci_free(py_dci_inst);

	cudaFree(data);
	cudaFree(query);
	cudaFree(final_outputs);
	cudaFree(final_distances);
}

int main() {
	small_scale(1);
	return 0;
}