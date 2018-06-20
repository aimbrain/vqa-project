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

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from layers import NeighbourhoodGraphConvolution as GraphConvolution
from layers import GraphLearner


class Model(nn.Module):

    def __init__(self,
                 vocab_size,
                 emb_dim,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 pretrained_wemb,
                 dropout,
                 n_kernels=8,
                 neighbourhood_size=16):

        '''
        ## Variables:
        - vocab_size: dimensionality of input vocabulary
        - emb_dim: question embedding size
        - feat_dim: dimensionality of input image features
        - out_dim: dimensionality of the output
        - dropout: dropout probability
        - n_kernels : number of Gaussian kernels for convolutions
        - bias: whether to add a bias to Gaussian kernels
        '''

        super(Model, self).__init__()

        # Set parameters
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.neighbourhood_size = neighbourhood_size

        # initialize word embedding layer weight
        self.wembed = nn.Embedding(vocab_size, emb_dim)
        self.wembed.weight.data.copy_(torch.from_numpy(pretrained_wemb))

        # question encoding
        self.q_lstm = nn.GRU(input_size=emb_dim, hidden_size=hid_dim)

        # graph learner
        self.adjacency_1 = GraphLearner(in_feature_dim=feat_dim + hid_dim,
                                        combined_feature_dim=512,
                                        K=36,
                                        dropout=dropout)

        # dropout layers
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_q = nn.Dropout(p=dropout/2)

        # graph convolution layers
        self.graph_convolution_1 = \
            GraphConvolution(feat_dim, hid_dim * 2, n_kernels, 2)
        self.graph_convolution_2 = \
            GraphConvolution(hid_dim * 2, hid_dim, n_kernels, 2)

        # output classifier
        self.out_1 = nn.utils.weight_norm(nn.Linear(hid_dim, out_dim))
        self.out_2 = nn.utils.weight_norm(nn.Linear(out_dim, out_dim))

    def forward(self, question, image, K, qlen):
        '''
        ## Inputs:
        - question (batch_size, max_qlen): input tokenised question
        - image (batch_size, K, feat_dim): input image features
        - K (int): number of image features/objects in the image
        - qlen (batch_size): vector describing the length (in words) of each input question
        ## Returns:
        - logits (batch_size, out_dim)
        '''

        K = int(K[0].cpu().data.numpy())

        # extract bounding boxes and compute centres
        bb = image[:, :, -4:].contiguous()
        bb_size = (bb[:, :, 2:]-bb[:, :, :2])
        bb_centre = bb[:, :, :2] + 0.5*bb_size

        # apply dropout to image features
        image = self.dropout(image)

        # Compute pseudo coordinates
        pseudo_coord = self._compute_pseudo(bb_centre)

        # Compute question encoding
        emb = self.wembed(question)
        packed = pack_padded_sequence(emb, qlen, batch_first=True)  # questions have variable lengths
        _, hid = self.q_lstm(packed)
        qenc = hid[0].unsqueeze(1)
        qenc_repeat = qenc.repeat(1, K, 1)

        # Learn adjacency matrix
        image_qenc_cat = torch.cat((image, qenc_repeat), dim=-1)
        adjacency_matrix = self.adjacency_1(image_qenc_cat)

        # Graph convolution 1
        neighbourhood_image, neighbourhood_pseudo = self._create_neighbourhood(image,
                                                                               pseudo_coord,
                                                                               adjacency_matrix,
                                                                               self.neighbourhood_size,
                                                                               weight=True)
        hidden_graph_1 = self.graph_convolution_1(
            neighbourhood_image, neighbourhood_pseudo)
        hidden_graph_1 = F.relu(hidden_graph_1)
        hidden_graph_1 = self.dropout(hidden_graph_1)

        # graph convolution 2
        hidden_graph_1, neighbourhood_pseudo = self._create_neighbourhood(hidden_graph_1,
                                                                          pseudo_coord,
                                                                          adjacency_matrix,
                                                                          self.neighbourhood_size,
                                                                          weight=False)
        hidden_graph_2 = self.graph_convolution_2(
            hidden_graph_1, neighbourhood_pseudo)
        hidden_graph_2 = F.relu(hidden_graph_2)

        hidden_graph_2, _ = torch.max(hidden_graph_2, dim=1)
        h = F.relu(qenc).squeeze(1)*hidden_graph_2

        # Output classifier
        hidden_1 = self.out_1(h)
        hidden_1 = F.relu(hidden_1)
        hidden_1 = self.dropout(hidden_1)
        logits = self.out_2(hidden_1)

        return logits, adjacency_matrix

    def _create_neighbourhood_feat(self, image, top_ind):
        '''
        ## Inputs:
        - image (batch_size, K, feat_dim)
        - top_ind (batch_size, K, neighbourhood_size)
        ## Returns:
        - neighbourhood_image (batch_size, K, neighbourhood_size, feat_dim)
        '''

        batch_size = image.size(0)
        K = image.size(1)
        feat_dim = image.size(2)
        neighbourhood_size = top_ind.size(-1)
        image = image.unsqueeze(1).expand(batch_size, K, K, feat_dim)
        idx = top_ind.unsqueeze(-1).expand(batch_size,
                                           K, neighbourhood_size, feat_dim)
        return torch.gather(image, dim=2, index=idx)

    def _create_neighbourhood_pseudo(self, pseudo, top_ind):
        '''
        ## Inputs:
        - pseudo_coord (batch_size, K, K, coord_dim)
        - top_ind (batch_size, K, neighbourhood_size)
        ## Returns:
        - neighbourhood_pseudo (batch_size, K, neighbourhood_size, coord_dim)
        '''
        batch_size = pseudo.size(0)
        K = pseudo.size(1)
        coord_dim = pseudo.size(3)
        neighbourhood_size = top_ind.size(-1)
        idx = top_ind.unsqueeze(-1).expand(batch_size,
                                           K, neighbourhood_size, coord_dim)
        return torch.gather(pseudo, dim=2, index=idx)

    def _create_neighbourhood(self,
                              features,
                              pseudo_coord,
                              adjacency_matrix,
                              neighbourhood_size,
                              weight=True):

        '''

        Creates a neighbourhood system for each graph node/image object

        ## Inputs:
        - features (batch_size, K, feat_dim): input image features
        - pseudo_coord (batch_size, K, K, coord_dim): pseudo coordinates for graph convolutions
        - adjacency_matrix (batch_size, K, K): learned adjacency matrix
        - neighbourhood_size (int)
        - weight (bool): specify if the features should be weighted by the adjacency matrix values

        ## Returns:
        - neighbourhood_image (batch_size, K, neighbourhood_size, feat_dim)
        - neighbourhood_pseudo (batch_size, K, neighbourhood_size, coord_dim)
        '''

        # Number of graph nodes
        K = features.size(1)

        # extract top k neighbours for each node and normalise
        top_k, top_ind = torch.topk(
            adjacency_matrix, k=neighbourhood_size, dim=-1, sorted=False)
        top_k = torch.stack([F.softmax(top_k[:, k]) for k in range(K)]).transpose(0, 1)  # (batch_size, K, neighbourhood_size)

        # extract top k features and pseudo coordinates
        neighbourhood_image = \
            self._create_neighbourhood_feat(features, top_ind)
        neighbourhood_pseudo = \
            self._create_neighbourhood_pseudo(pseudo_coord, top_ind)

        # weight neighbourhood features with graph edge weights
        if weight:
            neighbourhood_image = top_k.unsqueeze(-1)*neighbourhood_image

        return neighbourhood_image, neighbourhood_pseudo

    def _compute_pseudo(self, bb_centre):
        '''

        Computes pseudo-coordinates from bounding box centre coordinates

        ## Inputs:
        - bb_centre (batch_size, K, coord_dim)
        - polar (bool: polar or euclidean coordinates)
        ## Returns:
        - pseudo_coord (batch_size, K, K, coord_dim)
        '''

        K = bb_centre.size(1)

        # Compute cartesian coordinates (batch_size, K, K, 2)
        pseudo_coord = bb_centre.view(-1, K, 1, 2) - \
            bb_centre.view(-1, 1, K, 2)

        # Conver to polar coordinates
        rho = torch.sqrt(
            pseudo_coord[:, :, :, 0]**2 + pseudo_coord[:, :, :, 1]**2)
        theta = torch.atan2(
            pseudo_coord[:, :, :, 0], pseudo_coord[:, :, :, 1])
        pseudo_coord = torch.cat(
            (torch.unsqueeze(rho, -1), torch.unsqueeze(theta, -1)), dim=-1)

        return pseudo_coord
