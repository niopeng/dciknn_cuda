/*
 * Code for Fast k-Nearest Neighbour Search via Prioritized DCI
 *
 * This code implements the method described by Li et al., which can be found at https://arxiv.org/abs/1703.00440
 * This code also builds off of code written by Ke Li.
 */

#ifndef DCI_H
#define DCI_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdbool.h>


typedef struct idx_elem {
	float key;  // value of the projection of point onto vector
	int value;  // index of the point
} idx_elem;

// sorting alg we are using
__device__
void mix_sort(idx_elem arr[], int n);

float compute_dist(const float* const vec1, const float* const vec2,
		const int dim);

typedef struct dci {
	int dim;                // (Ambient) dimensionality of data
	int num_comp_indices;   // Number of composite indices
	int num_simp_indices;   // Number of simple indices in each composite index
	int num_points;
	idx_elem* indices; // Assuming row-major layout, matrix of size required_num_points x (num_comp_indices*num_simp_indices)
	float* proj_vec; // Assuming row-major layout, matrix of size dim x (num_comp_indices*num_simp_indices)
	float* data_proj;    // Device copy of data_proj
	float* data;
	float* d_data;
	int devID;              // To initialize CUDA's matmul, set to 0
} dci;

typedef struct dci_query_config {
	bool blind;
	int num_outer_iterations;
	int max_num_candidates;
} dci_query_config;

void dci_gen_proj_vec(float* proj_vec, const int dim,
		const int num_indices);

void dci_init(dci* const dci_inst, const int dim, const int num_comp_indices,
		const int num_simp_indices);

__device__
void insertion_sort(idx_elem arr[], int n);

// // Note: the data itself is not kept in the index and must be kept in-place
void dci_add(dci* const dci_inst, const int dim, const int num_points,
		float* const data, const int block_size, const int thread_size);

void dci_query(dci* const dci_inst, const int dim, const int num_queries,
		const float* const query, const int num_neighbours,
		const dci_query_config query_config, int* const nearest_neighbours,
		float* const nearest_neighbour_dists, const int block_size,
		const int thread_size);

void dci_clear(dci* const dci_inst);

// Clear indices and reset the projection directions
void dci_reset(dci* const dci_inst);

void dci_free(const dci* const dci_inst);

void dci_dump(const dci* const dci_inst);

#endif // DCI_H
