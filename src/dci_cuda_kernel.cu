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

#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "dci.h"
#include "util.h"

/* Sorting functions */
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

/* CUDA runtime */
#include <cuda_runtime.h>
#include <cublas_v2.h>

__device__
float compute_dist_device(const float* const vec1, const float* const vec2,
		const int dim) {
	int i;
	float sq_dist = 0.0;
	for (i = 0; i < dim; i++) {
		sq_dist += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
	}
	return sqrt(sq_dist);
}

__device__
static inline float abs_d(float x) {
	return x > 0 ? x : -x;
}

/* Normalize the input projection vectors. Vectors are normalized along each row. */
__global__ void normalize_proj_vecs(float* const proj_vec, const int dim,
		const int num_indices, const int num_heads) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	/* Note: Assumes num_blocks = num_threads */
	int total_indices = num_indices * num_heads;
	int chunk_size = (total_indices + blockDim.x * blockDim.x - 1)
			/ (blockDim.x * blockDim.x);
	int vec_index;
	for (int j = 0; j < chunk_size; ++j) {
		vec_index = i * chunk_size + j;
		if (vec_index < total_indices) {
			float sq_norm = 0.0;
			for (int k = 0; k < dim; ++k) {
				sq_norm += proj_vec[vec_index * dim + k]
						* proj_vec[vec_index * dim + k];
			}
			float norm = sqrtf(sq_norm);
			for (int k = 0; k < dim; ++k) {
				proj_vec[vec_index * dim + k] /= norm;
			}
		}
	}
}

/* Create matrix with proj_vec dim-dimensional normalized gaussian vectors.
 vectors are normalized along each row */
void dci_gen_proj_vec(float* const proj_vec, const int dim,
		const int num_indices, const int num_heads) {
	/* Generate the random indices */
	rng_parallel_device(proj_vec, dim * num_indices * num_heads, GAUSS_RAND);

	/* Normalize */
	int block_size = 32;
	int thread_size = 32;
	normalize_proj_vecs<<<block_size, thread_size>>>(proj_vec, dim,
			num_indices, num_heads);

	/* Synchronize the threads */
	cudaDeviceSynchronize();
}

/* Initializes the master DCI data structure.  */
void dci_init(dci* const dci_inst, const int dim, const int num_heads, const int num_comp_indices,
		const int num_simp_indices, const int devId) {

	int num_indices = num_comp_indices * num_simp_indices;

	dci_inst->dim = dim;
	dci_inst->num_heads = num_heads;
	dci_inst->num_comp_indices = num_comp_indices;
	dci_inst->num_simp_indices = num_simp_indices;

	cudaMallocManaged((void **) &dci_inst->proj_vec,
			sizeof(float) * dim * num_indices * num_heads);
	dci_gen_proj_vec(dci_inst->proj_vec, dim, num_indices, num_heads);

	// testing: same project vector for every head
	/*
	for (int h = 0; h < num_heads; h++) {
		for (int i = 0; i < dim * num_indices; i++) {
			dci_inst->proj_vec[i + dim * num_indices * h] = dci_inst->proj_vec[i];
		}
	}
	*/

	/* Variables that initialize to default values */
	dci_inst->num_points = 0;
	dci_inst->indices = NULL;
	dci_inst->data = NULL;
	dci_inst->devID = devId;
}

/* Sort index within each head  
	each block work on same portion of index just that different thread within each block work on differnt head. 
	ex. block 0 work on indices 0-100, thread 5 work on indices for head 1 */
__global__ void sort_indices(dci* const dci_inst, const int num_indices, const int num_heads,
		const int num_points, const int points_per_block) {

	// number of thread assign to a single head
	// ex. 10 thread, 2 head, then 5 thread / head
	int blockDim_head = (int) (blockDim.x / num_heads);
	
	int total = num_indices;
	int chunk_size = (total + blockDim_head - 1) / blockDim_head;
	int curr_head = (int) (threadIdx.x / blockDim_head);
	int curr_head_thread = threadIdx.x % blockDim_head;

	int idx;
	int num_points_in_block = min(
			(int) (dci_inst->num_points - blockIdx.x * points_per_block),
			points_per_block);
	for (int j = 0; j < chunk_size; j++) {
		idx = curr_head_thread * chunk_size + j; // position within each head
		if (idx < total) {
			mix_sort(
					&(dci_inst->indices[idx * dci_inst->num_points
							+ blockIdx.x * points_per_block 
							+ num_points * num_indices * curr_head]), // adjust position for different head
					num_points_in_block);
		}
	}
}

/* Copy data in proj_vec to indices */
__global__ void copy_to_indices(dci* const dci_inst, float* const data_proj,
		const int num_indices, const int num_points, const int num_heads) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int n = num_indices * num_points * num_heads;
	int chunk_size = (n + blockDim.x * gridDim.x - 1)
			/ (blockDim.x * gridDim.x);
	int idx;
	for (int j = 0; j < chunk_size; j++) {
		idx = i * chunk_size + j;
		if (idx < n) {
			dci_inst->indices[idx].key = data_proj[idx];
			dci_inst->indices[idx].value = (idx % num_points); // only consider the position in the current head
		}
	}
}

/* Add data to the master DCI data structure.  */
void dci_add(dci* const dci_inst, const int dim, const int num_points, const int num_heads,
		float* const data, const int block_size, const int thread_size) {

	int num_indices = dci_inst->num_comp_indices * dci_inst->num_simp_indices;
	float *data_proj;
	cudaMallocManaged((void **) &data_proj,
			sizeof(float) * num_points * num_indices * num_heads);

	assert(dim == dci_inst->dim);
	assert(num_heads == dci_inst->num_heads);
	assert(dci_inst->num_points == 0);

	cudaMallocManaged((void **) &dci_inst->data,
			sizeof(float) * num_points * dim * num_heads);
	dci_inst->data = data;
	cudaMallocManaged((void **) &dci_inst->indices,
			sizeof(idx_elem) * num_points * num_indices * num_heads);

	dci_inst->num_points = num_points;

	for (int i = 0; i < num_heads; i++) {
		int proj_vec_id = i * dim * num_indices;
		int data_id = i * num_points * dim;
		int data_proj_id = i * num_points * num_indices;
		matmul_device(
			CUBLAS_OP_N, 
			CUBLAS_OP_T, 
			num_indices, 
			num_points,
			dci_inst->dim,
			&(dci_inst->proj_vec[proj_vec_id]), 
			&(dci_inst->data[data_id]), 
			&(data_proj[data_proj_id]), 
			dci_inst->devID
		);
	}
	cudaDeviceSynchronize();

	/* Add to indices */
	copy_to_indices	<<<block_size, thread_size>>>(dci_inst, data_proj, num_indices, num_points, num_heads);

	/* Synchronize the threads */
	cudaDeviceSynchronize();

	int points_per_block = (dci_inst->num_points + block_size - 1) / block_size;
	/* Sort the indices */
	sort_indices<<<block_size, thread_size>>>(dci_inst, num_indices, num_heads, num_points, points_per_block);

	/* Synchronize the threads */
	cudaDeviceSynchronize();

	cudaFree(data_proj);
}

__device__
void insertion_sort(idx_elem arr[], int n) {
	int i, j;
	idx_elem key;
	for (i = 1; i < n; i++) {
		key = arr[i];
		j = i - 1;
		while (j >= 0 && arr[j].key > key.key) {
			arr[j + 1] = arr[j];
			j = j - 1;
		}
		arr[j + 1] = key;
	}
}

/* Modified quick_sort to use "mix_sort" below. */
__device__
void quick_sort(idx_elem arr[], int n) {
	// arbitrary pivot
	float pivot_key = arr[n / 2].key;
	idx_elem swp;
	int low = 0;
	int high = n - 1;
	while (low < n || high > 0) {
		while (arr[low].key < pivot_key && low < n) {
			low++;
		}
		while (arr[high].key > pivot_key && high > 0) {
			high--;
		}
		if (low <= high) {
			swp = arr[low];
			arr[low] = arr[high];
			arr[high] = swp;
			low++;
			high--;
		} else {
			if (high > 0) {
				mix_sort(arr, high + 1);
			}
			if (low < n - 1) {
				mix_sort(&arr[low], n - low);
			}
			return;
		}
	}
}

/* Sorting algorithm. If the number of data points is fewer than 64, then it does
 Insertion Sort. Otherwise, it uses Quick Sort. The reasoning is that if there are
 too few data points, then Quick Sort's overhead may be too large. */
__device__
void mix_sort(idx_elem arr[], int n) {
	if (n > 64) {
		quick_sort(arr, n);
	} else {
		insertion_sort(arr, n);
	}
}

/* find the next closest query projection
	blockDim_head represent the number of thread assign to each head
 */
__device__
static inline int dci_next_closest_proj(const idx_elem* const idx,
		int* const left_pos, int* const right_pos, const float query_proj,
		const int num_elems, const int blockDim_head) {
	int cur_pos;
	int lower_bound = -blockDim_head;
	int upper_bound = num_elems + blockDim_head - 1;
	if ((*left_pos <= lower_bound) && (*right_pos >= upper_bound)) {
		cur_pos = lower_bound;
	} else if (*left_pos <= lower_bound) {
		cur_pos = *right_pos;
		(*right_pos) += blockDim_head;
	} else if (*right_pos >= upper_bound) {
		cur_pos = *left_pos;
		(*left_pos) -= blockDim_head;
	} else if (idx[min(*right_pos, num_elems - 1)].key - query_proj
			< query_proj - idx[max(*left_pos, 0)].key) {
		cur_pos = *right_pos;
		(*right_pos) += blockDim_head;
	} else {
		cur_pos = *left_pos;
		(*left_pos) -= blockDim_head;
	}
	return cur_pos;
}

/* Returns the index of the element whose key is the largest that is less than the key
	Returns an integer from -1 to num_elems - 1 inclusive
	Could return -1 if all elements are greater or equal to key */
__device__
static inline int dci_search_index(const idx_elem* const idx, const float key,
		const int num_elems) {
	int start_pos, end_pos, cur_pos;

	start_pos = -1;
	end_pos = num_elems - 1;
	cur_pos = (start_pos + end_pos + 2) / 2;

	while (start_pos < end_pos) {
		if (idx[cur_pos].key < key) {
			start_pos = cur_pos;
		} else {
			end_pos = cur_pos - 1;
		}
		cur_pos = (start_pos + end_pos + 2) / 2;
	}

	return start_pos;
}

/* search index
	blockDim_head represent the number of thread assign to each head
 */
__device__ void search_index(const dci* const dci_inst, const float* const query_proj_column, 
		const int num_indices, const int num_heads, 
		int* const left_pos, int* const right_pos, 
		const int points_per_block, const int blockDim_head) {

	int total = num_indices;
	int chunk_size = (total + blockDim_head - 1) / blockDim_head;
	int curr_head = (int) (threadIdx.x / blockDim_head);
	int curr_head_thread = threadIdx.x % blockDim_head;

	int idx;
	for (int j = 0; j < chunk_size; j++) {
		idx = curr_head_thread * chunk_size + j; // idx of indices within each head
		if (idx < total) {
			left_pos[idx + curr_head * num_indices] = dci_search_index(
				&(dci_inst->indices[idx * dci_inst->num_points
						+ blockIdx.x * points_per_block
						+ dci_inst->num_points * num_indices * curr_head]), // adjust index for each head
				query_proj_column[idx + curr_head * num_indices], // adjust index for each head
				min(dci_inst->num_points - blockIdx.x * points_per_block,
						points_per_block)) - blockDim_head + 1;

			right_pos[idx + curr_head * num_indices] = 
				left_pos[idx + curr_head * num_indices] + blockDim_head;
		}
	}
}

/* assign index priority to each index
*/
__device__ void init_index_priority(const dci* const dci_inst,
		const float* const query_proj_column, 
		const int num_indices, const int num_heads, 
		int* const left_pos, int* const right_pos, float* const index_priority,
		int* const cur_pos, const int points_per_block, const int blockDim_head) {

	int total = num_indices;
	int chunk_size = (total + blockDim_head - 1) / blockDim_head;
	int num_points_in_block = min(
			(int) (dci_inst->num_points - blockIdx.x * points_per_block),
			points_per_block);

	int curr_head = (int) (threadIdx.x / blockDim_head);
	int curr_head_thread = threadIdx.x % blockDim_head;
	
	int idx;
	for (int j = 0; j < chunk_size; j++) {
		idx = curr_head_thread * chunk_size + j; // idx of indices within each head
		if (idx < total && num_points_in_block > 0) {
			cur_pos[idx + curr_head * num_indices] = dci_next_closest_proj(
				&(dci_inst->indices[idx * (dci_inst->num_points) // find the current index within each head
					+ blockIdx.x * points_per_block // get the point that is processed by this block
					+ dci_inst->num_points * num_indices * curr_head]), // start of the head / adjust for current head
				&(left_pos[idx + curr_head * num_indices]), 
				&(right_pos[idx + curr_head * num_indices]),
				query_proj_column[idx + curr_head * num_indices],
				num_points_in_block, blockDim_head);

			int position;
			if ((cur_pos[idx + curr_head * num_indices] < 0) && (cur_pos[idx + curr_head * num_indices] > -blockDim_head)) {
				position = 0;
			} else if ((cur_pos[idx + curr_head * num_indices] < (num_points_in_block + blockDim_head - 1))
					&& (cur_pos[idx + curr_head * num_indices] >= num_points_in_block)) {
				position = num_points_in_block - 1;
			} else {
				position = cur_pos[idx + curr_head * num_indices];
			}

			assert(position >= 0); // There should be at least one point in the index
			assert(position < num_points_in_block);
			index_priority[idx + curr_head * num_indices] = abs_d(
					dci_inst->indices[position + idx * (dci_inst->num_points)	// position of index (single head)
						+ blockIdx.x * points_per_block // position within each index
						+ dci_inst->num_points * num_indices * curr_head].key
							- query_proj_column[idx + curr_head * num_indices]);
		}
	}
}

__global__ void init_counts(const dci* const dci_inst, int* counts) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int total = dci_inst->num_comp_indices * dci_inst->num_points;
	total = dci_inst->num_heads * total;
	int chunk_size = (total + blockDim.x * gridDim.x - 1)
			/ (blockDim.x * gridDim.x);
	for (int j = 0; j < chunk_size; j++) {
		int l = i * chunk_size + j;
		if (l < total) {
			counts[l] = 0;
		}
	}
}

__global__ void init_candidate_dists(const dci* const dci_inst,
		float* candidate_dists) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int total = dci_inst->num_points * dci_inst->num_heads;
	int chunk_size = (total + blockDim.x * gridDim.x - 1)
			/ (blockDim.x * gridDim.x);
	for (int j = 0; j < chunk_size; j++) {
		int l = i * chunk_size + j;
		if (l < total) {
			candidate_dists[l] = -2.0;
		}
	}
}

__global__ void init_candidate_indices(const dci* const dci_inst,
		int* candidate_indices) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int total = dci_inst->num_points;
	int chunk_size = (total + blockDim.x * gridDim.x - 1)
			/ (blockDim.x * gridDim.x);
	for (int j = 0; j < chunk_size; j++) {
		int l = i * chunk_size + j;
		if (l < total) {
			candidate_indices[l] = -1;
		}
	}
}

// Blind querying does not compute distances or look at the values of indexed vectors
// For blind querying, top_candidates is not used; all_candidates is used to store candidates in the order of retrieval
__global__
static void dci_query_single_point_by_block(const dci* const dci_inst,
		const int num_neighbours, const int num_queries, 
		const float* const query, const float* const query_proj_column,
		const dci_query_config query_config, float* const d_top_candidates_dist, 
		int* const d_top_candidates_index, int* const all_candidates, 
		int* counts, float* candidate_dists) {

	int j, h;
	float cur_dist;
	int num_indices = dci_inst->num_comp_indices * dci_inst->num_simp_indices;
	int num_heads = dci_inst->num_heads;

	float last_top_candidate_dist = -1.0; // The distance of the k^th closest candidate found so far
	int num_candidates = 0, last_top_candidate = -1;

	int max_possible_num_candidates = min(
			query_config.max_num_candidates,
			query_config.num_outer_iterations);

	int blockDim_head = (int) (blockDim.x / num_heads);
	int curr_head = (int) (threadIdx.x / blockDim_head);

	// for each head there are a number of thread assign to each head, and head_threadIdx is just thread id adjust to head
	int head_threadIdx = threadIdx.x % blockDim_head;

	int points_per_block = (dci_inst->num_points + gridDim.x - 1) / gridDim.x; // default number of data processed by a block
	int num_points_in_block = min(
			(int) (dci_inst->num_points - blockIdx.x * points_per_block), // should not process data beyond the total number of data
			points_per_block);

	if (num_points_in_block > 0) {
		// shared value is an array, each value in the array is correspond to a head
		// the array size is num_heads, thread work on same head has same shared value
		__shared__ int could_break_all, k, m;
		__shared__ float *top_index_priority;
		__shared__ int *top_h;
		__shared__ int *position;
		__shared__ int *i;
		__shared__ bool *could_break; // Bug fix: resolve infinite loop if thread 0 exits first

		__shared__ int* left_pos;
		__shared__ int* right_pos;
		__shared__ int* cur_pos;
		__shared__ float* index_priority;

		// init variables
		if (threadIdx.x == 0) {
			top_index_priority = new float[num_heads];
			top_h = new int[num_heads];
			position = new int[num_heads];
			i = new int[num_heads];
			could_break =new bool[num_heads];

			left_pos = new int[num_indices * num_heads];
			right_pos = new int[num_indices * num_heads];
			cur_pos = new int[num_indices * num_heads];
			index_priority = new float[num_indices * num_heads];

			could_break[curr_head] = false;
			could_break_all = 0;
			k = 0;
		}

		__syncthreads();

		// left_pos and right_pos already account for multi-head
		search_index(
			dci_inst, 
			query_proj_column, 
			num_indices, 
			num_heads,
			left_pos,
			right_pos,
			points_per_block,
			blockDim_head
		);

		__syncthreads();

		init_index_priority(
			dci_inst, 
			query_proj_column, 
			num_indices, 
			num_heads,
			left_pos, 
			right_pos,
			index_priority, 
			cur_pos, 
			points_per_block,
			blockDim_head
		);

		__syncthreads();

		while (k < num_points_in_block * dci_inst->num_simp_indices * blockDim_head) {

			if (threadIdx.x == 0) {
				m = 0;
			}
			__syncthreads();

			// iterate for each complex index (work properly)
			while (m < dci_inst->num_comp_indices) {
				// first thread only
				// For each complex index, we find the simple index that has the lowest
				// index priority, that is cloest to the query point (projection on projection 
				// vector), this simple index will be top_h

				// inner loop one
				if ((threadIdx.x % blockDim_head) == 0) {
					// Get the top priority and data index in priority queue
					top_index_priority[curr_head] = DBL_MAX;
					top_h[curr_head] = -1;
					for (h = 0; h < dci_inst->num_simp_indices; h++) {
						if (index_priority[h + m * dci_inst->num_simp_indices + curr_head * num_indices]
								< top_index_priority[curr_head]) {
							top_index_priority[curr_head] = index_priority[h 
								+ m * dci_inst->num_simp_indices 
								+ curr_head * num_indices];
							top_h[curr_head] = h;
						}
					}
				}

				__syncthreads();

				if (top_h[curr_head] >= 0) {
					if ((threadIdx.x % blockDim_head) == 0) {
						i[curr_head] = top_h[curr_head] + m * dci_inst->num_simp_indices + curr_head * num_indices;
						position[curr_head] = cur_pos[i[curr_head]];
					}
				}

				__syncthreads();

				if (top_h[curr_head] >= 0) {
					int cur_index = position[curr_head] + head_threadIdx;

					if (cur_index >= 0 && cur_index < num_points_in_block) {
						int cur_point = dci_inst->indices[cur_index
								+ dci_inst->num_points * i[curr_head]
								+ blockIdx.x * points_per_block].value; // cur_point is index within the head (i[curr_head] already adjust to head)

						counts[cur_point + dci_inst->num_points * m
							+ dci_inst->num_comp_indices * dci_inst->num_points * curr_head]++;

						if (counts[cur_point + dci_inst->num_points * m
								+ dci_inst->num_comp_indices * dci_inst->num_points * curr_head]
								== dci_inst->num_simp_indices) { 
						
							// add offset to candidate_dists
							if (candidate_dists[cur_point + dci_inst->num_points * curr_head] == -2.0) {
								if (query_config.blind) {
									candidate_dists[cur_point + dci_inst->num_points * curr_head] = -1.0;
									// lock
									all_candidates[num_candidates
											+ blockIdx.x * max_possible_num_candidates
											+ max_possible_num_candidates * gridDim.x * curr_head] =
											cur_point;
									num_candidates++;		
								} else {
									// Compute distance
									cur_dist = compute_dist_device(
											&(dci_inst->data[cur_point * dci_inst->dim
													+ dci_inst->num_points * dci_inst->dim * curr_head]), 
											&(query[dci_inst->dim * num_queries * curr_head]), dci_inst->dim);

									candidate_dists[cur_point + dci_inst->num_points * curr_head] = cur_dist;
									if (num_candidates < num_neighbours) {
										d_top_candidates_dist[blockIdx.x * num_neighbours
												+ head_threadIdx * num_neighbours
												+ num_candidates
												+ gridDim.x * blockDim.x * num_neighbours * curr_head] = cur_dist;
										d_top_candidates_index[blockIdx.x * num_neighbours
												+ head_threadIdx * num_neighbours
												+ num_candidates
												+ gridDim.x * blockDim.x * num_neighbours * curr_head] = cur_point;
										if (cur_dist > last_top_candidate_dist) {
											last_top_candidate_dist = cur_dist;
											last_top_candidate = num_candidates;
										}
									} else if (cur_dist < last_top_candidate_dist) {
										d_top_candidates_dist[blockIdx.x * num_neighbours
												+ head_threadIdx * num_neighbours
												+ last_top_candidate
												+ gridDim.x * blockDim.x * num_neighbours * curr_head] = cur_dist;
										d_top_candidates_index[blockIdx.x * num_neighbours // each block has only certain number of neighbours
												+ head_threadIdx * num_neighbours
												+ last_top_candidate
												+ gridDim.x * blockDim.x * num_neighbours * curr_head] = cur_point; // adjust by head
										last_top_candidate_dist = -1.0;
										// Assuming num_neighbours less than the min(blockDim) = 32
										// no need to run on gpu
										for (j = 0; j < num_neighbours; j++) {
											if (d_top_candidates_dist[blockIdx.x * num_neighbours
													+ head_threadIdx * num_neighbours
													+ j
													+ gridDim.x * blockDim.x * num_neighbours * curr_head]
													> last_top_candidate_dist) {
												last_top_candidate_dist =
														d_top_candidates_dist[blockIdx.x * num_neighbours
																+ head_threadIdx * num_neighbours
																+ j
																+ gridDim.x * blockDim.x * num_neighbours * curr_head];
												last_top_candidate = j;
											}
										}
									}
									num_candidates++;
								}
							} else {
								if (!query_config.blind) {
									cur_dist = candidate_dists[cur_point + dci_inst->num_points * curr_head];
								}
							}
						}
						
					}
				}

				__syncthreads();

				if (top_h[curr_head] >= 0) {
					// use the first thread work on each head to update
					if ((threadIdx.x % blockDim_head) == 0) {

						cur_pos[i[curr_head]] = dci_next_closest_proj(
								&(dci_inst->indices[i[curr_head] * (dci_inst->num_points)
										+ blockIdx.x * points_per_block]),
								&(left_pos[i[curr_head]]), &(right_pos[i[curr_head]]), query_proj_column[i[curr_head]], // need reconsider
								num_points_in_block, blockDim_head);

						if ((cur_pos[i[curr_head]] < 0) && (cur_pos[i[curr_head]] > -blockDim_head)) {
							position[curr_head] = 0;
						} else if ((cur_pos[i[curr_head]]
								< (num_points_in_block + blockDim_head - 1))
								&& (cur_pos[i[curr_head]] >= num_points_in_block)) {
							position[curr_head] = num_points_in_block - 1;
						} else {
							position[curr_head] = cur_pos[i[curr_head]];
						}

						if (position[curr_head] >= 0 && position[curr_head] < num_points_in_block) {
							index_priority[i[curr_head]] = abs_d(
									dci_inst->indices[position[curr_head]
											+ i[curr_head] * (dci_inst->num_points)
											+ blockIdx.x * points_per_block].key
											- query_proj_column[i[curr_head]]);
						} else {
							index_priority[i[curr_head]] = DBL_MAX;
							cur_pos[i[curr_head]] = -blockDim_head;
						}
					}
				}

				if (threadIdx.x == 0) {
					m = m + 1;
				}
				__syncthreads();
			}

			if ((threadIdx.x % blockDim_head) == 0) {
				if (!could_break[curr_head]) {
					if (num_candidates >= num_neighbours) {
						if (k + 1 >= query_config.num_outer_iterations
										* dci_inst->num_simp_indices
								|| num_candidates >= query_config.max_num_candidates) {
							could_break[curr_head] = true;
							could_break_all++;
						}
					}
				}
			}

			if (threadIdx.x == 0) {
				k = k + 1;
			}

			__syncthreads();

			// need to ensure all could_break is could break 
			if (could_break_all == num_heads) {
			    break;
			}
		}

		__syncthreads();

		// free variables
		if (threadIdx.x == 0) {
			free(left_pos);
			free(right_pos);
			free(cur_pos);
			free(index_priority);

			free(top_index_priority); 
			free(top_h);
			free(position);
			free(i);
			free(could_break);
		}
	}
}

__global__ void mix_sort_kernel(idx_elem* const d_top_candidates,
		const int total) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		mix_sort(d_top_candidates, total);
	}
}

__global__ void update_top(const dci* const dci_inst,
		double* const index_priority, int const comp_index, int* top_h,
		int *mutex) {
	double top_h_priority = DBL_MAX;
	//	Shared top priority array
	extern __shared__ double top_priority[];
	//	Shared top priority index in data array
	extern __shared__ double top_index[];

	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;
	top_priority[tid] = DBL_MAX;
	top_index[tid] = idx % dci_inst->num_simp_indices;

	while (idx < dci_inst->num_simp_indices) {
		double cur_priority = index_priority[comp_index
				* dci_inst->num_simp_indices + idx];
		if (top_priority[tid] > cur_priority) {
			top_priority[tid] = cur_priority;
			top_index[tid] = idx % dci_inst->num_simp_indices;
		}
		idx += gridDim.x * blockDim.x;
	}
	__syncthreads();
	idx = blockIdx.x * blockDim.x + tid;
	// block-wide reduction
	for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
		if (tid < offset && idx < dci_inst->num_simp_indices) {
			double cur_priority = index_priority[comp_index
					* dci_inst->num_simp_indices + tid];
			double compare_priority = index_priority[comp_index
					* dci_inst->num_simp_indices + tid + offset];
			if (cur_priority > compare_priority) {
				top_priority[tid] = compare_priority;
				top_index[tid] = (blockIdx.x * blockDim.x + tid + offset)
						% dci_inst->num_simp_indices;
			}
		}
		__syncthreads();
	}

	// finally, thread 0 writes the result
	if (threadIdx.x == 0) {
		while (atomicCAS(mutex, 0, 1) != 0)
			;  //lock
		if (top_priority[0] < top_h_priority) {
			top_h_priority = top_priority[0];
			*top_h = top_index[0];
		}
		atomicExch(mutex, 0);  //unlock
	}
}

/*
 * Update the top nearest neighbors with distance from the partial results
 */
void get_top_candidates(int* const nearest_neighbours,
		float* const nearest_neighbour_dists,
		float* const d_top_candidates_dist, int* const d_top_candidates_index,
		const int num_neighbours, const int total) {
	thrust::sort_by_key(thrust::device, d_top_candidates_dist,
			d_top_candidates_dist + total, d_top_candidates_index);
	cudaMemcpy(nearest_neighbour_dists, d_top_candidates_dist,
			sizeof(float) * num_neighbours, cudaMemcpyDeviceToDevice);
	cudaMemcpy(nearest_neighbours, d_top_candidates_index,
			sizeof(int) * num_neighbours, cudaMemcpyDeviceToDevice);
}

__global__ void init_dist(float* const candidate_map, const int total,
		const float value) {
	int idx, i = blockDim.x * blockIdx.x + threadIdx.x;
	int chunk_size = (total + blockDim.x * gridDim.x - 1)
			/ (blockDim.x * gridDim.x);
	int j;
	// initialize the counters
	for (j = 0; j < chunk_size; j++) {
		idx = i * chunk_size + j;
		if (idx < total) {
			candidate_map[idx] = value;
		}
	}
}

__global__ void init_candidates(idx_elem* const candidate_map, const int total,
		const float value) {
	int idx, i = blockDim.x * blockIdx.x + threadIdx.x;
	int chunk_size = (total + blockDim.x * gridDim.x - 1)
			/ (blockDim.x * gridDim.x);
	int j;
	// initialize the counters
	for (j = 0; j < chunk_size; j++) {
		idx = i * chunk_size + j;
		if (idx < total) {
			candidate_map[idx].key = value;
			candidate_map[idx].value = -1;
		}
	}
}

__global__ void get_blind_candidate_count(idx_elem* const candidate_map,
		int* const d_all_candidates, const int total, 
		const int num_points, const int num_indices, const int num_heads) {
	int curr_head;
	int idx, i = blockDim.x * blockIdx.x + threadIdx.x;
	int chunk_size = (total * num_heads + blockDim.x * gridDim.x - 1)
			/ (blockDim.x * gridDim.x);
	int j;
	// maintain counts as negative numbers for candidate_map.key in order to reuse mix_sort (ascending)
	for (j = 0; j < chunk_size; j++) {
		idx = i * chunk_size + j;
		curr_head = (int) (idx / total);	// which head the given index belong to

		if (idx < total) {
			candidate_map[d_all_candidates[idx]].key--;
			candidate_map[d_all_candidates[idx]].value = 
					d_all_candidates[idx];
		}
	}
}

/*
 * Update the top nearest neighbors from the partial results
 * total = block_size * max_possible_num_candidates
 */
void get_top_blind_candidates(int* const nearest_neighbours,
		int* const d_all_candidates, const int max_possible_num_candidates,
		const int num_points, const int num_indices,
		const int num_neighbours, const int num_queries, const int num_heads,
		const int total) {
	int i, j;
	idx_elem* candidate_map;
	cudaMallocManaged((void **) (&candidate_map),
			sizeof(idx_elem) * total * num_heads);
	int block_size = 1024;
	int thread_size = 32;
	init_candidates<<<block_size, thread_size>>>(candidate_map, total * num_heads, 0);
	// synch all blocks
	cudaDeviceSynchronize();
	get_blind_candidate_count<<<block_size, thread_size>>>(candidate_map, d_all_candidates, total, num_points, num_indices, num_heads);
	// synch all blocks
	cudaDeviceSynchronize();

	for (j = 0; j < num_heads; j++) {
		mix_sort_kernel<<<1, 1>>>(&(candidate_map[max_possible_num_candidates * block_size * j]), total);
		
		for (i = 0; i < max_possible_num_candidates; i++) {
			nearest_neighbours[i + num_neighbours * num_queries * j] = 
				candidate_map[i + num_neighbours * num_queries * j].value;
		}
	}
}

// change the dimension of query project from (head, query, indices) to (query, head, indices)
__global__ void dci_query_proj_3d_permute(float* const query_proj, float* const query_proj_column, 
		const int num_heads, const int num_queries, const int num_indices) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int total = num_heads * num_queries;
	int chunk_size = (total + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);

	int idx, head, query;
	for (int j = 0; j < chunk_size; j++) {
		idx = i * chunk_size + j;

		if (idx < total) { // prevent overflow
			head = (int) (idx / num_queries);
			query = idx % num_queries;
			for (int k = 0; k < num_indices; k++) {
				query_proj_column[query * num_heads * num_indices + head * num_indices + k] =
					query_proj[head * num_queries * num_indices + query * num_indices + k];
			}
		}
	}
}

// If blind querying is used, nearest_neighbours must be of size num_queries * max_possible_num_candidates; otherwise, it must be of size num_queries * num_neighbours
// nearest_neighbour_dists can be NULL when blind querying is used
void dci_query(dci* const dci_inst, const int dim, const int num_heads, const int num_queries,
		const float* const query, const int num_neighbours,
		const dci_query_config query_config, int* const nearest_neighbours,
		float* const nearest_neighbour_dists, const int block_size,
		const int thread_size) {

	int num_indices = dci_inst->num_comp_indices * dci_inst->num_simp_indices;
	int max_possible_num_candidates = min(query_config.max_num_candidates,
			query_config.num_outer_iterations);

	assert(dim == dci_inst->dim);
	assert(num_neighbours > 0);
	assert(num_neighbours <= dci_inst->num_points);

	// for fixing timeout
	void* dummy;
	cudaMalloc(&dummy, 1);

	// calculate query_proj
	int devId = 0;
	float* query_proj;
	float* query_proj_column;

	cudaMallocManaged((void **) (&query_proj),
			sizeof(float) * num_indices * num_queries * num_heads);

	cudaMallocManaged((void **) (&query_proj_column),
			sizeof(float) * num_indices * num_queries * num_heads);		

	for (int i = 0; i < num_heads; i++) {
		int query_id = i * dci_inst->dim * num_queries;
		int proj_vec_id = i * dci_inst->dim * num_indices;
		int query_proj_id = i * num_indices * num_queries;

		matmul_device(
			CUBLAS_OP_N, 
			CUBLAS_OP_T, 
			num_queries, 
			num_indices,
			dci_inst->dim,
			&(query[query_id]), 
			&(dci_inst->proj_vec[proj_vec_id]), 
			&(query_proj[query_proj_id]), 
			devId
		);
	}
	cudaDeviceSynchronize();

	dci_query_proj_3d_permute<<<block_size, thread_size>>>(query_proj, query_proj_column, num_heads, num_queries, num_indices);
	cudaDeviceSynchronize();

	// copy query config to device pointer
	dci_query_config* d_query_config;
	cudaMallocManaged((void **) (&d_query_config),
			sizeof(dci_query_config));
	cudaMemcpy(d_query_config, &query_config, sizeof(dci_query_config),
			cudaMemcpyHostToDevice);

	// make the raw nearest neighbors
	int* d_all_candidates;
	cudaMallocManaged((void **) (&d_all_candidates),
			sizeof(int) * max_possible_num_candidates * block_size * num_heads);

	float* d_top_candidates_dist;
	cudaMalloc((void **) (&d_top_candidates_dist),
			sizeof(float) * num_neighbours * block_size * thread_size * num_heads);
	int* d_top_candidates_index;
	cudaMalloc((void **) (&d_top_candidates_index),
			sizeof(int) * num_neighbours * block_size * thread_size * num_heads);

	int* counts;
	cudaMallocManaged((void **) (&counts),
			sizeof(int) * dci_inst->num_points * dci_inst->num_comp_indices * num_heads);

	float* candidate_dists;
	cudaMallocManaged((void **) (&candidate_dists),
			sizeof(float) * dci_inst->num_points * num_heads);

	for (int j = 0; j < num_queries; j++) { 

		// need to refresh the result holder to avoid carry over results
		init_dist<<<block_size, thread_size>>>(d_top_candidates_dist,
				num_neighbours * block_size * thread_size * num_heads, DBL_MAX);

		cudaDeviceSynchronize();
		init_counts<<<block_size, thread_size>>>(dci_inst, counts);
		init_candidate_dists<<<block_size, thread_size>>>(dci_inst,
				candidate_dists);

		cudaDeviceSynchronize();

		dci_query_single_point_by_block<<<block_size, thread_size>>>(
				dci_inst,
				num_neighbours, 
				num_queries,
				&(query[j * dim]), 
				&(query_proj_column[j * num_indices * num_heads]),
				*d_query_config,
				d_top_candidates_dist, 
				d_top_candidates_index, 
				d_all_candidates,
				counts, 
				candidate_dists
			);

		cudaDeviceSynchronize();

		// output
		if (!query_config.blind) {
			for (int h = 0; h < num_heads; h++) {
				get_top_candidates(
						&(nearest_neighbours[j * num_neighbours + num_neighbours * num_queries * h]),
						&(nearest_neighbour_dists[j * num_neighbours + num_neighbours * num_queries * h]),
						&(d_top_candidates_dist[num_neighbours * block_size * thread_size * h]), 
						&(d_top_candidates_index[num_neighbours * block_size * thread_size * h]),
						num_neighbours, 
						block_size * num_neighbours * thread_size
					);
			}
		} else {
			get_top_blind_candidates(
					&(nearest_neighbours[j * max_possible_num_candidates]),
					d_all_candidates, 
					max_possible_num_candidates,
					dci_inst->num_points,
					num_indices,
					num_neighbours,
					num_queries,
					num_heads,
					block_size * max_possible_num_candidates
				);
		}
	}

	// free the allocated memories
	cudaFree(query_proj);
	cudaFree(query_proj_column);
	cudaFree(d_query_config);
	cudaFree(d_all_candidates);
	cudaFree(d_top_candidates_dist);
	cudaFree(d_top_candidates_index);
	cudaFree(counts);
	cudaFree(candidate_dists);
}


void dci_clear(dci* const dci_inst) {
	if (dci_inst->indices) {
		cudaFree(dci_inst->indices);
		dci_inst->indices = NULL;
	}
	dci_inst->data = NULL;
	dci_inst->num_points = 0;
}

void dci_reset(dci* const dci_inst) {
	dci_clear(dci_inst);
	dci_gen_proj_vec(dci_inst->proj_vec, dci_inst->dim,
			dci_inst->num_comp_indices * dci_inst->num_simp_indices, dci_inst->num_heads);
}

void dci_free(const dci* const dci_inst) {
	if (dci_inst->indices) {
		cudaFree(dci_inst->indices);
	}
	cudaFree(dci_inst->proj_vec);

}

void dci_dump(const dci* const dci_inst) {
	int i, j;
	int num_indices = dci_inst->num_comp_indices * dci_inst->num_simp_indices;
	for (j = 0; j < num_indices; j++) {
		for (i = 0; i < dci_inst->num_points; i++) {
			printf("%f[%d],",
					dci_inst->indices[i + j * (dci_inst->num_points)].key,
					dci_inst->indices[i + j * (dci_inst->num_points)].value);
		}
		printf("\n");
	}
}
