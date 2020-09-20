/*
 * Code for Fast k-Nearest Neighbour Search via Prioritized DCI
 *
 * This code implements the method described by Li et al., which can be found at https://arxiv.org/abs/1703.00440
 * This code also builds off of code written by Ke Li.
 */

#include <malloc.h>
#include <stdlib.h>
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
		const int num_indices) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	/* Note: Assumes num_blocks = num_threads */
	int chunk_size = (num_indices + blockDim.x * blockDim.x - 1)
			/ (blockDim.x * blockDim.x);
	int vec_index;
	for (int j = 0; j < chunk_size; ++j) {
		vec_index = i * chunk_size + j;
		if (vec_index < num_indices) {
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
		const int num_indices) {
	/* Generate the random indices */
	rng_parallel_device(proj_vec, dim * num_indices, GAUSS_RAND);

	/* Normalize */
	int block_size = 32;
	int thread_size = 32;
	normalize_proj_vecs<<<block_size, thread_size>>>(proj_vec, dim,
			num_indices);

	/* Synchronize the threads */
	cudaDeviceSynchronize();
}

	/* Initializes the master DCI data structure.  */
void dci_init(dci* const dci_inst, const int dim, const int num_comp_indices,
		const int num_simp_indices) {
	int num_indices = num_comp_indices * num_simp_indices;

	dci_inst->dim = dim;
	dci_inst->num_comp_indices = num_comp_indices;
	dci_inst->num_simp_indices = num_simp_indices;

	cudaMallocManaged((void **) &dci_inst->proj_vec,
			sizeof(float) * dim * num_indices);
	dci_gen_proj_vec(dci_inst->proj_vec, dim, num_indices);

	/* Variables that initialize to default values */
	dci_inst->num_points = 0;
	dci_inst->indices = NULL;
	dci_inst->data = NULL;
	dci_inst->devID = 0;
}

/* Sort indices */
__global__ void sort_indices(dci* const dci_inst, const int num_indices,
		const int num_points, const int points_per_block) {
	int chunk_size = (num_indices + blockDim.x - 1) / blockDim.x;
	int index;
	int num_points_in_block = min(
			(int) (dci_inst->num_points - blockIdx.x * points_per_block),
			points_per_block);
	for (int j = 0; j < chunk_size; j++) {
		index = threadIdx.x * chunk_size + j;
		if (index < num_indices) {
			mixSort(
					&(dci_inst->indices[index * dci_inst->num_points
							+ points_per_block * blockIdx.x]),
					num_points_in_block);
		}
	}
}

/* Copy data in proj_vec to indices */
__global__ void copy_to_indices(dci* const dci_inst, float* const data_proj,
		const int num_indices, const int num_points) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int n = num_indices * num_points;
	int chunk_size = (n + blockDim.x * gridDim.x - 1)
			/ (blockDim.x * gridDim.x);
	int index;
	for (int j = 0; j < chunk_size; j++) {
		index = i * chunk_size + j;
		if (index < n) {
			dci_inst->indices[index].key = data_proj[index];
			dci_inst->indices[index].value = index % num_points;
		}
	}
}

/* Add data to the master DCI data structure.  */
void dci_add(dci* const dci_inst, const int dim, const int num_points,
		float* const data, const int block_size, const int thread_size) {
	int num_indices = dci_inst->num_comp_indices * dci_inst->num_simp_indices;
	float *data_proj;
	cudaMallocManaged((void **) &data_proj,
			sizeof(float) * num_points * num_indices);

	assert(dim == dci_inst->dim);
	assert(dci_inst->num_points == 0);

	cudaMallocManaged((void **) &dci_inst->data,
			sizeof(float) * num_points * dim);
	dci_inst->data = data;
	cudaMallocManaged((void **) &dci_inst->indices,
			sizeof(idx_elem) * num_points * num_indices);

	dci_inst->num_points = num_points;

	matmul_device(CUBLAS_OP_N, CUBLAS_OP_T, num_indices, num_points,
			dci_inst->dim, dci_inst->proj_vec, dci_inst->data, data_proj,
			dci_inst->devID);
	cudaDeviceSynchronize();

	/* Add to indices */
	copy_to_indices	<<<block_size, thread_size>>>(dci_inst, data_proj, num_indices, num_points);

	/* Synchronize the threads */
	cudaDeviceSynchronize();

	int points_per_block = (dci_inst->num_points + block_size - 1) / block_size;
	/* Sort the indices */
	sort_indices<<<block_size, thread_size>>>(dci_inst, num_indices, num_points,
			points_per_block);

	/* Synchronize the threads */
	cudaDeviceSynchronize();

	cudaFree(data_proj);
}

/* Insertion sort. Taken from StackOverflow */
__device__
void insertionSort(idx_elem arr[], int n) {
	int i, j;
	idx_elem key;
	for (i = 1; i < n; i++) {
		key = arr[i];
		j = i - 1;

		/* Move elements of arr[0..i-1], that are
		 greater than key, to one position ahead
		 of their current position */
		while (j >= 0 && arr[j].key > key.key) {
			arr[j + 1] = arr[j];
			j = j - 1;
		}
		arr[j + 1] = key;
	}
}

/* Modified Quicksort to use "MixSort" below. */
__device__
void quickSort(idx_elem arr[], int n) {
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
				mixSort(arr, high + 1);
			}
			if (low < n - 1) {
				mixSort(&arr[low], n - low);
			}
			return;
		}
	}
}

/* Sorting algorithm. If the number of data points is fewer than 64, then it does
 Insertion Sort. Otherwise, it uses Quick Sort. The reasoning is that if there are
 too few data points, then Quick Sort's overhead may be too large. */
__device__
void mixSort(idx_elem arr[], int n) {
	if (n > 64) {
		quickSort(arr, n);
	} else {
		insertionSort(arr, n);
	}
}

__device__
static inline int dci_next_closest_proj(const idx_elem* const index,
		int* const left_pos, int* const right_pos, const float query_proj,
		const int num_elems) {
	int cur_pos;
	int lower_bound = -blockDim.x;
	int upper_bound = num_elems + blockDim.x - 1;
	if ((*left_pos <= lower_bound) && (*right_pos >= upper_bound)) {
		cur_pos = lower_bound;
	} else if (*left_pos <= lower_bound) {
		cur_pos = *right_pos;
		(*right_pos) += blockDim.x;
	} else if (*right_pos >= upper_bound) {
		cur_pos = *left_pos;
		(*left_pos) -= blockDim.x;
	} else if (index[min(*right_pos, num_elems - 1)].key - query_proj
			< query_proj - index[max(*left_pos, 0)].key) {
		cur_pos = *right_pos;
		(*right_pos) += blockDim.x;
	} else {
		cur_pos = *left_pos;
		(*left_pos) -= blockDim.x;
	}
	return cur_pos;
}

// Returns the index of the element whose key is the largest that is less than the key
// Returns an integer from -1 to num_elems - 1 inclusive
// Could return -1 if all elements are greater or equal to key
__device__
static inline int dci_search_index(const idx_elem* const index, const float key,
		const int num_elems) {
	int start_pos, end_pos, cur_pos;

	start_pos = -1;
	end_pos = num_elems - 1;
	cur_pos = (start_pos + end_pos + 2) / 2;

	while (start_pos < end_pos) {
		if (index[cur_pos].key < key) {
			start_pos = cur_pos;
		} else {
			end_pos = cur_pos - 1;
		}
		cur_pos = (start_pos + end_pos + 2) / 2;
	}

	return start_pos;
}

/* Search indices */
__device__ void search_index(const dci* const dci_inst,
		const float* const query_proj, const int num_indices,
		int* const left_pos, int* const right_pos, const int points_per_block) {
	int total = num_indices;
	int chunk_size = (total + blockDim.x - 1) / blockDim.x;
	int index;
	for (int j = 0; j < chunk_size; j++) {
		index = threadIdx.x * chunk_size + j;
		if (index < total) {
			left_pos[index] = dci_search_index(
					&(dci_inst->indices[index * (dci_inst->num_points)
							+ blockIdx.x * points_per_block]),
					query_proj[index],
					min(dci_inst->num_points - blockIdx.x * points_per_block,
							points_per_block)) - blockDim.x + 1;
			right_pos[index] = left_pos[index] + blockDim.x;
		}
	}
}

/* Find the closest point at index */
__device__ void find_closest(const dci* const dci_inst,
		const float* const query_proj, const int num_indices,
		int* const left_pos, int* const right_pos, float* const index_priority,
		int* const cur_pos, const int points_per_block) {
	int total = num_indices;
	int chunk_size = (total + blockDim.x - 1) / blockDim.x;
	int index;
	int num_points_in_block = min(
			(int) (dci_inst->num_points - blockIdx.x * points_per_block),
			points_per_block);
	for (int j = 0; j < chunk_size; j++) {
		index = threadIdx.x * chunk_size + j;
		if (index < total && num_points_in_block > 0) {
			cur_pos[index] = dci_next_closest_proj(
					&(dci_inst->indices[index * (dci_inst->num_points)
							+ blockIdx.x * points_per_block]),
					&(left_pos[index]), &(right_pos[index]), query_proj[index],
					num_points_in_block);
			int position;
			if ((cur_pos[index] < 0) && (cur_pos[index] > -blockDim.x)) {
				position = 0;
			} else if ((cur_pos[index] < (num_points_in_block + blockDim.x - 1))
					&& (cur_pos[index] >= num_points_in_block)) {
				position = num_points_in_block - 1;
			} else {
				position = cur_pos[index];
			}
			assert(position >= 0); // There should be at least one point in the index
			assert(position < num_points_in_block);
			index_priority[index] = abs_d(
					dci_inst->indices[position + index * (dci_inst->num_points)
							+ blockIdx.x * points_per_block].key
							- query_proj[index]);
		}
	}
}

__global__ void init_counts(const dci* const dci_inst, int* counts) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int total = dci_inst->num_comp_indices * dci_inst->num_points;
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
	int total = dci_inst->num_points;
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
		const int num_neighbours, const float* const query,
		const float* const query_proj, const dci_query_config query_config,
		float* const d_top_candidates_dist, int* const d_top_candidates_index,
		int* const all_candidates, int* counts, float* candidate_dists) {
	int j, h;
	float cur_dist;
	int num_indices = dci_inst->num_comp_indices * dci_inst->num_simp_indices;
	__shared__ float top_index_priority;
	__shared__ int k, top_h, position, m, i;
	float last_top_candidate_dist = -1.0; // The distance of the k^th closest candidate found so far
	int num_candidates = 0, last_top_candidate = -1;

	// init variables
	if (threadIdx.x == 0) {
		k = 0;
	}

	int max_possible_num_candidates = min(query_config.max_num_candidates,
			query_config.num_outer_iterations);

	int points_per_block = (dci_inst->num_points + gridDim.x - 1) / gridDim.x;
	int num_points_in_block = min(
			(int) (dci_inst->num_points - blockIdx.x * points_per_block),
			points_per_block);

	if (num_points_in_block > 0) {

		__shared__ int* left_pos;
		__shared__ int* right_pos;
		__shared__ int* cur_pos;
		__shared__ float* index_priority;
		// init variables
		if (threadIdx.x == 0) {
			left_pos = new int[num_indices];
			right_pos = new int[num_indices];
			cur_pos = new int[num_indices];
			index_priority = new float[num_indices];
		}
		__syncthreads();

		/* Search index */
		search_index(dci_inst, query_proj, num_indices, left_pos, right_pos,
				points_per_block);

		/* Synchronize the threads */
		__syncthreads();

		/* Populate the closest indices */
		find_closest(dci_inst, query_proj, num_indices, left_pos, right_pos,
				index_priority, cur_pos, points_per_block);

		/* Synchronize the threads */
		__syncthreads();

		while (k < num_points_in_block * dci_inst->num_simp_indices * blockDim.x) {

			if (threadIdx.x == 0) {
				m = 0;
			}
			__syncthreads();
			while (m < dci_inst->num_comp_indices) {
				// only one thread to get the top
				if (threadIdx.x == 0) {
					/* Get the top priority and data index in priority queue */
					top_index_priority = DBL_MAX;
					top_h = -1;
					for (h = 0; h < dci_inst->num_simp_indices; h++) {
						if (index_priority[h + m * dci_inst->num_simp_indices]
								< top_index_priority) {
							top_index_priority = index_priority[h
									+ m * dci_inst->num_simp_indices];
							top_h = h;
						}
					}
				}
				/* Synchronize the threads */
				__syncthreads();
				if (top_h >= 0) {
					if (threadIdx.x == 0) {
						i = top_h + m * dci_inst->num_simp_indices;
						position = cur_pos[i];
					}
					__syncthreads();
					int cur_index = position + threadIdx.x;
					// check whether the current thread pointing index is within range
					if (cur_index >= 0 && cur_index < num_points_in_block) {
						int cur_point = dci_inst->indices[cur_index
								+ i * (dci_inst->num_points)
								+ blockIdx.x * points_per_block].value;
						counts[cur_point + m * (dci_inst->num_points)]++;
						if (counts[cur_point + m * (dci_inst->num_points)]
								== dci_inst->num_simp_indices) {
							// add offset to candidate_dists
							if (candidate_dists[cur_point] == -2.0) {
								if (query_config.blind) {
									candidate_dists[cur_point] = -1.0;
									// lock
									all_candidates[num_candidates
											+ blockIdx.x
													* max_possible_num_candidates] =
											cur_point;
									num_candidates++;
								} else {
									// Compute distance
									cur_dist = compute_dist_device(
											&(dci_inst->data[cur_point
													* dci_inst->dim]), query,
											dci_inst->dim);
									candidate_dists[cur_point] = cur_dist;
									if (num_candidates < num_neighbours) {
										d_top_candidates_dist[blockIdx.x
												* num_neighbours
												+ threadIdx.x * num_neighbours
												+ num_candidates] = cur_dist;
										d_top_candidates_index[blockIdx.x
												* num_neighbours
												+ threadIdx.x * num_neighbours
												+ num_candidates] = cur_point;
										if (cur_dist > last_top_candidate_dist) {
											last_top_candidate_dist = cur_dist;
											last_top_candidate = num_candidates;
										}
									} else if (cur_dist < last_top_candidate_dist) {
										d_top_candidates_dist[blockIdx.x
												* num_neighbours
												+ threadIdx.x * num_neighbours
												+ last_top_candidate] = cur_dist;
										d_top_candidates_index[blockIdx.x
												* num_neighbours
												+ threadIdx.x * num_neighbours
												+ last_top_candidate] = cur_point;
										last_top_candidate_dist = -1.0;
										// Assuming num_neighbours less than the min(blockDim) = 32
										// no need to run on gpu
										for (j = 0; j < num_neighbours; j++) {
											if (d_top_candidates_dist[blockIdx.x
													* num_neighbours
													+ threadIdx.x * num_neighbours
													+ j]
													> last_top_candidate_dist) {
												last_top_candidate_dist =
														d_top_candidates_dist[blockIdx.x
																* num_neighbours
																+ threadIdx.x
																		* num_neighbours
																+ j];
												last_top_candidate = j;
											}
										}
									}
									num_candidates++;
								}
							} else {
								if (!query_config.blind) {
									cur_dist = candidate_dists[cur_point];
								}
							}
						}
					}
					/* Synchronize the threads */
					__syncthreads();
					// use the first thread to update
					if (threadIdx.x == 0) {
						cur_pos[i] = dci_next_closest_proj(
								&(dci_inst->indices[i * (dci_inst->num_points)
										+ blockIdx.x * points_per_block]),
								&(left_pos[i]), &(right_pos[i]), query_proj[i],
								num_points_in_block);
						if ((cur_pos[i] < 0) && (cur_pos[i] > -blockDim.x)) {
							position = 0;
						} else if ((cur_pos[i]
								< (num_points_in_block + blockDim.x - 1))
								&& (cur_pos[i] >= num_points_in_block)) {
							position = num_points_in_block - 1;
						} else {
							position = cur_pos[i];
						}
						if (position >= 0 && position < num_points_in_block) {
							index_priority[i] = abs_d(
									dci_inst->indices[position
											+ i * (dci_inst->num_points)
											+ blockIdx.x * points_per_block].key
											- query_proj[i]);
						} else {
							index_priority[i] = DBL_MAX;
							cur_pos[i] = -blockDim.x;
						}
					}
				}

				// __syncthreads();
				if (threadIdx.x == 0) {
					m++;
				}
				__syncthreads();
			}
			if (threadIdx.x == 0) {
				if (num_candidates >= num_neighbours) {
					if (k + 1
							>= query_config.num_outer_iterations
									* dci_inst->num_simp_indices
							|| num_candidates >= query_config.max_num_candidates) {
						break;
					}
				}
				k++;
			}
			/* Synchronize the threads */
			__syncthreads();
		}
		// free variables
		if (threadIdx.x == 0) {
			free(left_pos);
			free(right_pos);
			free(cur_pos);
			free(index_priority);
		}
	}


}

__global__ void mixSort_kernel(idx_elem* const d_top_candidates,
		const int total) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		mixSort(d_top_candidates, total);
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
	unsigned int index = blockIdx.x * blockDim.x + tid;
	top_priority[tid] = DBL_MAX;
	top_index[tid] = index % dci_inst->num_simp_indices;

	while (index < dci_inst->num_simp_indices) {
		double cur_priority = index_priority[comp_index
				* dci_inst->num_simp_indices + index];
		if (top_priority[tid] > cur_priority) {
			top_priority[tid] = cur_priority;
			top_index[tid] = index % dci_inst->num_simp_indices;
		}
		index += gridDim.x * blockDim.x;
	}
	__syncthreads();
	index = blockIdx.x * blockDim.x + tid;
	// block-wide reduction
	for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
		if (tid < offset && index < dci_inst->num_simp_indices) {
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
	int index, i = blockDim.x * blockIdx.x + threadIdx.x;
	int chunk_size = (total + blockDim.x * gridDim.x - 1)
			/ (blockDim.x * gridDim.x);
	int j;
	// initialize the counters
	for (j = 0; j < chunk_size; j++) {
		index = i * chunk_size + j;
		if (index < total) {
			candidate_map[index] = value;
		}
	}
}

__global__ void init_candidates(idx_elem* const candidate_map, const int total,
		const float value) {
	int index, i = blockDim.x * blockIdx.x + threadIdx.x;
	int chunk_size = (total + blockDim.x * gridDim.x - 1)
			/ (blockDim.x * gridDim.x);
	int j;
	// initialize the counters
	for (j = 0; j < chunk_size; j++) {
		index = i * chunk_size + j;
		if (index < total) {
			candidate_map[index].key = value;
			candidate_map[index].value = -1;
		}
	}
}

__global__ void count_blind_candidates(idx_elem* const candidate_map,
		int* const d_all_candidates, const int total) {
	int index, i = blockDim.x * blockIdx.x + threadIdx.x;
	int chunk_size = (total + blockDim.x * gridDim.x - 1)
			/ (blockDim.x * gridDim.x);
	int j;
	// add the counts as negative number in order to utilize mixSort later on
	for (j = 0; j < chunk_size; j++) {
		index = i * chunk_size + j;
		if (index < total) {
			candidate_map[d_all_candidates[index]].key--;
			candidate_map[d_all_candidates[index]].value =
					d_all_candidates[index];
		}
	}
}

/*
 * Update the top nearest neighbors from the partial results
 */
void get_top_blind_candidates(int* const nearest_neighbours,
		int* const d_all_candidates, const int max_possible_num_candidates,
		const int total) {
	int i;
	idx_elem* candidate_map;
	cudaMallocManaged((void **) (&candidate_map),
			sizeof(idx_elem) * total);
	int block_size = 1024;
	int thread_size = 32;
	init_candidates<<<block_size, thread_size>>>(candidate_map, total, 0);
	// synch all blocks
	cudaDeviceSynchronize();
	count_blind_candidates<<<block_size, thread_size>>>(candidate_map, d_all_candidates, total);
	// synch all blocks
	cudaDeviceSynchronize();
	mixSort_kernel<<<1, 1>>>(candidate_map, total);
	for (i = 0; i < max_possible_num_candidates; i++) {
		nearest_neighbours[i] = candidate_map[i].value;
	}
}

// If blind querying is used, nearest_neighbours must be of size num_queries * max_possible_num_candidates; otherwise, it must be of size num_queries * num_neighbours
// nearest_neighbour_dists can be NULL when blind querying is used
void dci_query(dci* const dci_inst, const int dim, const int num_queries,
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

	cudaMallocManaged((void **) (&query_proj),
			sizeof(float) * num_indices * num_queries);

	matmul_device(CUBLAS_OP_N, CUBLAS_OP_T, num_queries, num_indices,
			dci_inst->dim, query, dci_inst->proj_vec, query_proj, devId);

	// copy query config to device pointer
	dci_query_config* d_query_config;
	cudaMallocManaged((void **) (&d_query_config),
			sizeof(dci_query_config));
	cudaMemcpy(d_query_config, &query_config, sizeof(dci_query_config),
			cudaMemcpyHostToDevice);

	// make the raw nearest neighbors
	int* d_all_candidates;
	cudaMallocManaged((void **) (&d_all_candidates),
			sizeof(int) * max_possible_num_candidates * block_size);

	float* d_top_candidates_dist;
	cudaMalloc((void **) (&d_top_candidates_dist),
			sizeof(float) * num_neighbours * block_size * thread_size);
	int* d_top_candidates_index;
	cudaMalloc((void **) (&d_top_candidates_index),
			sizeof(int) * num_neighbours * block_size * thread_size);

	int* counts;
	cudaMallocManaged((void **) (&counts),
			sizeof(int) * dci_inst->num_points
					* dci_inst->num_comp_indices);

	float* candidate_dists;
	cudaMallocManaged((void **) (&candidate_dists),
			sizeof(float) * dci_inst->num_points);

	for (int j = 0; j < num_queries; j++) {
		// need to refresh the result holder to avoid carry over results
		init_dist<<<block_size, thread_size>>>(d_top_candidates_dist,
				num_neighbours * block_size * thread_size, DBL_MAX);

		cudaDeviceSynchronize();
		init_counts<<<block_size, thread_size>>>(dci_inst, counts);
		init_candidate_dists<<<block_size, thread_size>>>(dci_inst,
				candidate_dists);

		cudaDeviceSynchronize();

		dci_query_single_point_by_block<<<block_size, thread_size>>>(dci_inst,
				num_neighbours, &(query[j * dim]),
				&(query_proj[j * num_indices]), *d_query_config,
				d_top_candidates_dist, d_top_candidates_index, d_all_candidates,
				counts, candidate_dists);

		cudaDeviceSynchronize();

		// get the final output
		if (!query_config.blind) {
			get_top_candidates(&(nearest_neighbours[j * num_neighbours]),
					&(nearest_neighbour_dists[j * num_neighbours]),
					d_top_candidates_dist, d_top_candidates_index,
					num_neighbours, block_size * num_neighbours * thread_size);
		} else {
			get_top_blind_candidates(
					&(nearest_neighbours[j * max_possible_num_candidates]),
					d_all_candidates, max_possible_num_candidates,
					block_size * max_possible_num_candidates);
		}
	}

	// free the allocated memories
	cudaFree(query_proj);
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
			dci_inst->num_comp_indices * dci_inst->num_simp_indices);
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
