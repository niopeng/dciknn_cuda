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
from dciknn_cuda import DCI
import torch


def gen_data(ambient_dim, intrinsic_dim, num_points):
    latent_data = torch.randn((num_points, intrinsic_dim))
    transformation = torch.randn((intrinsic_dim, ambient_dim))
    data = torch.matmul(latent_data, transformation)
    return data     # num_points x ambient_dim


def main():
    assert torch.cuda.is_available()
    device = torch.device('cuda')

    #############################################################################################################################################
    #                                                                                                                                           #
    # Data Generation Hyperparameters                                                                                                           #
    #                                                                                                                                           #
    #############################################################################################################################################
    dim = 1000
    num_pts = 1000
    num_queries = 2

    intrinsic_dim = 50
    data_and_queries = gen_data(dim, intrinsic_dim, num_pts + num_queries)

    data = data_and_queries[:num_pts, :].detach().clone().to(device)
    query = data_and_queries[num_pts:, :].detach().clone().to(device)

    #############################################################################################################################################
    #                                                                                                                                           #
    # Problem Hyperparameter                                                                                                                    #
    #                                                                                                                                           #
    #############################################################################################################################################
    num_neighbours = 10  # The k in k-NN

    #############################################################################################################################################
    #                                                                                                                                           #
    # DCI Hyperparameters                                                                                                                       #
    #                                                                                                                                           #
    #############################################################################################################################################
    block_size = 100
    thread_size = 10
    num_comp_indices = 2
    num_simp_indices = 10
    num_outer_iterations = 5000

    # initialize the DCI instance
    dci_db = DCI(dim, num_comp_indices, num_simp_indices, block_size, thread_size)
    # Add data
    dci_db.add(data)
    # Query
    indices, dists = dci_db.query(query, num_neighbours, num_outer_iterations)
    print("Nearest Indices:", indices)
    print("Indices Distances:", dists)
    dci_db.clear()


if __name__ == '__main__':
    main()
