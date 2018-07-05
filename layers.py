#    Copyright 2018 AimBrain Ltd.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class NeighbourhoodGraphConvolution(Module):
    '''
    Implementation of: https://arxiv.org/pdf/1611.08402.pdf where we consider
    a fixed sized neighbourhood of nodes for each feature
    '''

    def __init__(self,
                 in_feat_dim,
                 out_feat_dim,
                 n_kernels,
                 coordinate_dim,
                 bias=False):
        super(NeighbourhoodGraphConvolution, self).__init__()
        '''
        ## Variables:
        - in_feat_dim: dimensionality of input features
        - out_feat_dim: dimensionality of output features
        - n_kernels: number of Gaussian kernels to use
        - coordinate_dim : dimensionality of the pseudo coordinates
        - bias: whether to add a bias to convolutional kernels
        '''

        # Set parameters
        self.n_kernels = n_kernels
        self.coordinate_dim = coordinate_dim
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.bias = bias

        # Convolution filters weights
        self.conv_weights = nn.ModuleList([nn.Linear(
            in_feat_dim, out_feat_dim//n_kernels, bias=bias) for i in range(n_kernels)])

        # Parameters of the Gaussian kernels
        self.mean_rho = Parameter(torch.Tensor(n_kernels, 1))
        self.mean_theta = Parameter(torch.Tensor(n_kernels, 1))
        self.precision_rho = Parameter(torch.Tensor(n_kernels, 1))
        self.precision_theta = Parameter(torch.Tensor(n_kernels, 1))

        self.init_parameters()

    def init_parameters(self):
        # Initialise Gaussian parameters
        self.mean_theta.data.uniform_(-np.pi, np.pi)
        self.mean_rho.data.uniform_(0, 1.0)
        self.precision_theta.data.uniform_(0.0, 1.0)
        self.precision_rho.data.uniform_(0.0, 1.0)

    def forward(self, neighbourhood_features, neighbourhood_pseudo_coord):
        '''
        ## Inputs:
        - neighbourhood_features (batch_size, K, neighbourhood_size, in_feat_dim)
        - neighbourhood_pseudo_coord (batch_size, K, neighbourhood_size, coordinate_dim)
        ## Returns:
        - convolved_features (batch_size, K, neighbourhood_size, out_feat_dim)
        '''

        # set parameters
        batch_size = neighbourhood_features.size(0)
        K = neighbourhood_features.size(1)
        neighbourhood_size = neighbourhood_features.size(2)

        # compute pseudo coordinate kernel weights
        weights = self.get_gaussian_weights(neighbourhood_pseudo_coord)
        weights = weights.view(
            batch_size*K, neighbourhood_size, self.n_kernels)

        # compute convolved features
        neighbourhood_features = neighbourhood_features.view(
            batch_size*K, neighbourhood_size, -1)
        convolved_features = self.convolution(neighbourhood_features, weights)
        convolved_features = convolved_features.view(-1, K, self.out_feat_dim)

        return convolved_features

    def get_gaussian_weights(self, pseudo_coord):
        '''
        ## Inputs:
        - pseudo_coord (batch_size, K, K, pseudo_coord_dim)
        ## Returns:
        - weights (batch_size*K, neighbourhood_size, n_kernels)
        '''

        # compute rho weights
        diff = (pseudo_coord[:, :, :, 0].contiguous().view(-1, 1) - self.mean_rho.view(1, -1))**2
        weights_rho = torch.exp(-0.5 * diff /
                                (1e-14 + self.precision_rho.view(1, -1)**2))

        # compute theta weights
        first_angle = torch.abs(pseudo_coord[:, :, :, 1].contiguous().view(-1, 1) - self.mean_theta.view(1, -1))
        second_angle = torch.abs(2 * np.pi - first_angle)
        weights_theta = torch.exp(-0.5 * (torch.min(first_angle, second_angle)**2)
                                  / (1e-14 + self.precision_theta.view(1, -1)**2))

        weights = weights_rho * weights_theta
        weights[(weights != weights).detach()] = 0

        # normalise weights
        weights = weights / torch.sum(weights, dim=1, keepdim=True)

        return weights

    def convolution(self, neighbourhood, weights):
        '''
        ## Inputs:
        - neighbourhood (batch_size*K, neighbourhood_size, in_feat_dim)
        - weights (batch_size*K, neighbourhood_size, n_kernels)
        ## Returns:
        - convolved_features (batch_size*K, out_feat_dim)
        '''
        # patch operator
        weighted_neighbourhood = torch.bmm(
            weights.transpose(1, 2), neighbourhood)

        # convolutions
        weighted_neighbourhood = [self.conv_weights[i](weighted_neighbourhood[:, i]) for i in range(self.n_kernels)]
        convolved_features = torch.cat([i.unsqueeze(1) for i in weighted_neighbourhood], dim=1)
        convolved_features = convolved_features.view(-1, self.out_feat_dim)

        return convolved_features


class GraphLearner(Module):
    def __init__(self, in_feature_dim, combined_feature_dim, K, dropout=0.0):
        super(GraphLearner, self).__init__()

        '''
        ## Variables:
        - in_feature_dim: dimensionality of input features
        - combined_feature_dim: dimensionality of the joint hidden embedding
        - K: number of graph nodes/objects on the image
        '''

        # Parameters
        self.in_dim = in_feature_dim
        self.combined_dim = combined_feature_dim
        self.K = K

        # Embedding layers
        self.edge_layer_1 = nn.Linear(in_feature_dim, 
                                      combined_feature_dim)
        self.edge_layer_2 = nn.Linear(combined_feature_dim, 
                                      combined_feature_dim)

        # Regularisation
        self.dropout = nn.Dropout(p=dropout)
        self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)

    def forward(self, graph_nodes):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''

        graph_nodes = graph_nodes.view(-1, self.in_dim)

        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)

        # layer 2
        h = self.edge_layer_2(h)
        h = F.relu(h)

        # outer product
        h = h.view(-1, self.K, self.combined_dim)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))

        return adjacency_matrix

