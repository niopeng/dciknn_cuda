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

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "dci.h"
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>
#include <thread>
#include <future>
#include <vector>
#include <stdio.h>

typedef struct py_dci {
    dci dci_inst;
    PyObject *py_array;
} py_dci;

namespace py = pybind11;

static void py_dci_free_wrap(PyObject *py_dci_inst_wrapper) {

    py_dci *py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    const at::cuda::OptionalCUDAGuard device_guard(py_dci_inst->dci_inst.devID);

    if (py_dci_inst->py_array) {
        Py_DECREF(py_dci_inst->py_array);
    }

    dci_free(&(py_dci_inst->dci_inst));
    cudaFree(py_dci_inst);
}

static void py_tensor_free(PyObject *py_tensor_wrapper) {
    torch::Tensor *py_tensor = (torch::Tensor *)PyCapsule_GetPointer(py_tensor_wrapper, "py_tensor");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(*py_tensor));
    cudaFree(py_tensor);
}

py::handle py_dci_new(const int dim, const int num_heads, const int num_comp_indices,
    const int num_simp_indices, const int deviceId) {

    printf("py_dci_new in dci_cuda.cpp\n");

    const at::cuda::OptionalCUDAGuard device_guard(deviceId);
    py_dci *py_dci_inst;
    cudaMallocManaged((void **) &py_dci_inst, sizeof(py_dci));

    // initialize DCI instance
    dci_init(&(py_dci_inst->dci_inst), dim, num_heads, num_comp_indices, num_simp_indices, deviceId);

    // Returns new reference
    PyObject *py_dci_inst_wrapper = PyCapsule_New(py_dci_inst, "py_dci_inst", py_dci_free_wrap);
    return py_dci_inst_wrapper;
}

void py_dci_add(py::handle py_dci_inst_wrapper, const int dim, const int num_points, const int num_heads,
    torch::Tensor py_data, const int block_size, const int thread_size) {

    printf("py_dci_add in dci_cuda.cpp\n");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(py_data));

    PyObject *py_obj = py_dci_inst_wrapper.ptr();
    py_dci *py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_obj, "py_dci_inst");
    float* data = (float *)py_data.data_ptr();

    // add data to DCI instance
    dci_add(&(py_dci_inst->dci_inst), dim, num_points, num_heads, data, block_size, thread_size);

    PyObject *py_tensor_wrapper = PyCapsule_New(&py_data, "py_tensor", py_tensor_free);
    py_dci_inst->py_array = py_tensor_wrapper;
    Py_INCREF(py_tensor_wrapper);
}


torch::Tensor py_dci_query(py::handle py_dci_inst_wrapper, const int dim, const int num_heads, const int num_queries,
    torch::Tensor py_query, const int num_neighbours, const bool blind, 
    const int num_outer_iterations, const int max_num_candidates, const int block_size, const int thread_size) {

    printf("py_dci_query in dci_cuda.cpp\n");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(py_query));

    PyObject *py_obj = py_dci_inst_wrapper.ptr();
    py_dci *py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_obj, "py_dci_inst");

    // Assuming row-major layout, py_query->data is N x D, where N is the number of queries and D is the dimensionality
    float* query = (float *)py_query.data_ptr();

    dci_query_config query_config = {blind, num_outer_iterations, max_num_candidates};
    int*  final_outputs;
    float* final_distances;
    const int output_size = num_neighbours * num_queries * num_heads;
    cudaMalloc((void **) &(final_outputs), sizeof(int) * output_size);
    cudaMalloc((void **) &(final_distances), sizeof(float) * output_size);

    //void dci_query(dci* const dci_inst, const int dim, const int num_heads, const int num_queries,
	//	const float* const query, const float* const query_column, const int num_neighbours,
	//	const dci_query_config query_config, int* const nearest_neighbours,
	//	float* const nearest_neighbour_dists, const int block_size,
	//	const int thread_size)

    // query using DCI
    dci_query(&(py_dci_inst->dci_inst), dim, num_heads, num_queries, query, 
        num_neighbours, query_config, final_outputs, final_distances, block_size, 
        thread_size);

    /*
    auto options = torch::TensorOptions().device(torch::kCUDA);
    auto new_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor final_outputs_array = torch::from_blob(final_outputs, {output_size}, new_options);
    // convert to float tensor to concatenate with the computed distances
    torch::Tensor final = final_outputs_array.to(torch::kFloat32);

    torch::Tensor final_distances_array = torch::from_blob(final_distances, {output_size}, options);

    torch::Tensor final_result = torch::cat({ final, final_distances_array }, 0);
    */

    return py_query;
}

std::vector<torch::Tensor> py_dci_multi_query(std::vector<py::handle> py_dci_inst_wrapper, const int dim, const int num_heads, 
    const int num_queries, std::vector<torch::Tensor> py_query, std::vector<torch::Tensor> py_query_column, const int num_neighbours, 
    const bool blind, const int num_outer_iterations, const int max_num_candidates, const int block_size,
    const int thread_size) {
    std::vector<torch::Tensor> results;
    std::vector<std::future<torch::Tensor>> calcs;
    for (unsigned int i = 0; i < py_query.size(); i++) {
        calcs.push_back(std::async(py_dci_query, py_dci_inst_wrapper[i], dim, num_heads, num_queries,
            py_query[i], py_query_column[i], num_neighbours, blind, num_outer_iterations, max_num_candidates, 
            block_size, thread_size));
    }
    for (unsigned int i = 0; i < py_query.size(); i++) {
        results.push_back(calcs[i].get());
    }
    return results;
}

void py_dci_clear(py::handle py_dci_inst_wrapper) {
    PyObject *py_obj = py_dci_inst_wrapper.ptr();
    py_dci *py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_obj, "py_dci_inst");
    const at::cuda::OptionalCUDAGuard device_guard(py_dci_inst->dci_inst.devID);

    if (py_dci_inst->py_array) {
        Py_DECREF(py_dci_inst->py_array);
    }

    dci_clear(&(py_dci_inst->dci_inst));
    py_dci_inst->py_array = NULL;
}

void py_dci_reset(py::handle py_dci_inst_wrapper) {
    PyObject *py_obj = py_dci_inst_wrapper.ptr();
    py_dci *py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_obj, "py_dci_inst");
    const at::cuda::OptionalCUDAGuard device_guard(py_dci_inst->dci_inst.devID);

    if (py_dci_inst->py_array) {
        Py_DECREF(py_dci_inst->py_array);
    }

    dci_reset(&(py_dci_inst->dci_inst));
    py_dci_inst->py_array = NULL;
}

void py_dci_free(py::handle py_dci_inst_wrapper) {
    PyObject *py_obj = py_dci_inst_wrapper.ptr();
    py_dci *py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_obj, "py_dci_inst");
    const at::cuda::OptionalCUDAGuard device_guard(py_dci_inst->dci_inst.devID);

    if (py_dci_inst->py_array) {
        Py_DECREF(py_dci_inst->py_array);
    }

    dci_free(&(py_dci_inst->dci_inst));
    cudaFree(py_dci_inst);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_dci_new", &py_dci_new, "Create new DCI instance. (CUDA)");
    m.def("_dci_add", &py_dci_add, "Add data. (CUDA)");
    m.def("_dci_query", &py_dci_query, "Search for nearest neighbours. (CUDA)");
    m.def("_dci_clear", &py_dci_clear, "Clear DCI. (CUDA)");
    m.def("_dci_reset", &py_dci_reset, "Reset DCI. (CUDA)");
    m.def("_dci_free", &py_dci_free, "Free DCI. (CUDA)");
    m.def("_dci_multi_query", &py_dci_multi_query, "Search for nearest neighbours with multiple GPUs. (CUDA)");
}
