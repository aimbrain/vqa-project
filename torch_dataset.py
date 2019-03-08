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

from __future__ import absolute_import, division, print_function

import os
import json
import numpy as np
import zarr
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import dataloader

try:
    import cPickle as pickle
except:
    import pickle as pickle


def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return dataloader.default_collate(batch)


class VQA_Dataset(Dataset):

    def __init__(self, data_dir, emb_dim=300, train=True):

        # Set parameters
        self.data_dir = data_dir  # directory where the data is stored
        self.emb_dim = emb_dim    # question embedding dimension
        self.train = train        # train (True) or eval (False) mode
        self.seqlen = 14          # maximum question sequence length

        # Load training question dictionary
        q_dict = pickle.load(
            open(os.path.join(data_dir, 'train_q_dict.p'), 'rb'))
        self.q_itow = q_dict['itow']
        self.q_wtoi = q_dict['wtoi']
        self.q_words = len(self.q_itow) + 1

        # Load training answer dictionary
        a_dict = pickle.load(
            open(os.path.join(data_dir, 'train_a_dict.p'), 'rb'))
        self.a_itow = a_dict['itow']
        self.a_wtoi = a_dict['wtoi']
        self.n_answers = len(self.a_itow) + 1

        # Load image features and bounding boxes
        self.i_feat = zarr.open(os.path.join(
            data_dir, 'trainval.zarr'), mode='r')
        self.bbox = zarr.open(os.path.join(
            data_dir, 'trainval_boxes.zarr'), mode='r')
        self.sizes = pd.read_csv(os.path.join(
            data_dir, 'trainval_image_size.csv'))

        # Load questions
        if train:
            self.vqa = json.load(
                open(os.path.join(data_dir, 'vqa_train_final_3000.json')))
        else:
            self.vqa = json.load(
                open(os.path.join(data_dir, 'vqa_val_final_3000.json')))

        self.n_questions = len(self.vqa)

        print('Loading done')
        self.feat_dim = self.i_feat[list(self.i_feat.keys())[
            0]].shape[1] + 4  # + bbox
        self.init_pretrained_wemb(emb_dim)

    def init_pretrained_wemb(self, emb_dim):
        """
            From blog.keras.io
            Initialises words embeddings with pre-trained GLOVE embeddings
        """
        embeddings_index = {}
        f = open(os.path.join(self.data_dir, 'glove.6B.') +
                 str(emb_dim) + 'd.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs
        f.close()

        embedding_mat = np.zeros((self.q_words, emb_dim), dtype=np.float32)
        for word, i in self.q_wtoi.items():
            embedding_v = embeddings_index.get(word)
            if embedding_v is not None:
                embedding_mat[i] = embedding_v

        self.pretrained_wemb = embedding_mat

    def __len__(self):
        return self.n_questions

    def __getitem__(self, idx):

        # question sample
        qlen = len(self.vqa[idx]['question_toked'])
        q = [0] * 100
        for i, w in enumerate(self.vqa[idx]['question_toked']):
            try:
                q[i] = self.q_wtoi[w]
            except:
                q[i] = 0    # validation questions may contain unseen word

        # soft label answers
        a = np.zeros(self.n_answers, dtype=np.float32)
        for w, c in self.vqa[idx]['answers_w_scores']:
            try:
                a[self.a_wtoi[w]] = c
            except:
                continue

        # number of votes for each answer
        n_votes = np.zeros(self.n_answers, dtype=np.float32)
        for w, c in self.vqa[idx]['answers']:
            try:
                n_votes[self.a_wtoi[w]] = c
            except:
                continue

        # id of the question
        qid = self.vqa[idx]['question_id']

        # image sample
        iid = self.vqa[idx]['image_id']
        img = self.i_feat[str(iid)]
        bboxes = np.asarray(self.bbox[str(iid)])
        imsize = self.sizes[str(iid)]

        if np.logical_not(np.isfinite(img)).sum() > 0:
            raise ValueError

        # number of image objects
        k = 36

        # scale bounding boxes by image dimensions
        for i in range(k):
            bbox = bboxes[i]
            bbox[0] /= imsize[0]
            bbox[1] /= imsize[1]
            bbox[2] /= imsize[0]
            bbox[3] /= imsize[1]
            bboxes[i] = bbox

        # format variables
        q = np.asarray(q)
        a = np.asarray(a).reshape(-1)
        n_votes = np.asarray(n_votes).reshape(-1)
        qid = np.asarray(qid).reshape(-1)
        i = np.concatenate([img, bboxes], axis=1)
        k = np.asarray(k).reshape(1)

        return q, a, n_votes, qid, i, k, qlen


class VQA_Dataset_Test(Dataset):

    def __init__(self, data_dir, emb_dim=300, train=True):
        self.data_dir = data_dir
        self.emb_dim = emb_dim
        self.train = train
        self.seqlen = 14    # hard set based on paper

        q_dict = pickle.load(
            open(os.path.join(data_dir, 'train_q_dict.p'), 'rb'))
        self.q_itow = q_dict['itow']
        self.q_wtoi = q_dict['wtoi']
        self.q_words = len(self.q_itow) + 1

        a_dict = pickle.load(
            open(os.path.join(data_dir, 'train_a_dict.p'), 'rb'))
        self.a_itow = a_dict['itow']
        self.a_wtoi = a_dict['wtoi']
        self.n_answers = len(self.a_itow) + 1

        if train:
            self.vqa = json.load(open(os.path.join(data_dir, 'vqa_train_final_3000.json'))) + \
                json.load(
                    open(os.path.join(data_dir, 'vqa_val_final_3000.json')))
            self.i_feat = zarr.open(os.path.join(
                data_dir, 'trainval.zarr'), mode='r')
            self.bbox = zarr.open(os.path.join(
                data_dir, 'trainval_boxes.zarr'), mode='r')
            self.sizes = pd.read_csv(os.path.join(
                data_dir, 'trainval_image_size.csv'))
        else:
            self.vqa = json.load(
                open(os.path.join(data_dir, 'vqa_test_toked.json')))
            self.i_feat = zarr.open(os.path.join(
                data_dir, 'test.zarr'), mode='r')
            self.bbox = zarr.open(os.path.join(
                data_dir, 'test_boxes.zarr'), mode='r')
            self.sizes = pd.read_csv(os.path.join(
                data_dir, 'test_image_size.csv'))

        self.n_questions = len(self.vqa)

        print('Loading done')
        self.feat_dim = self.i_feat[list(self.i_feat.keys())[
            0]].shape[1] + 4  # + bbox
        self.init_pretrained_wemb(emb_dim)

    def init_pretrained_wemb(self, emb_dim):
        """From blog.keras.io"""
        embeddings_index = {}
        f = open(os.path.join(self.data_dir, 'glove.6B.') +
                 str(emb_dim) + 'd.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs
        f.close()

        embedding_mat = np.zeros((self.q_words, emb_dim), dtype=np.float32)
        for word, i in self.q_wtoi.items():
            embedding_v = embeddings_index.get(word)
            if embedding_v is not None:
                embedding_mat[i] = embedding_v

        self.pretrained_wemb = embedding_mat

    def __len__(self):
        return self.n_questions

    def __getitem__(self, idx):

        # question sample
        qlen = len(self.vqa[idx]['question_toked'])
        q = [0] * 100
        for i, w in enumerate(self.vqa[idx]['question_toked']):
            try:
                q[i] = self.q_wtoi[w]
            except:
                q[i] = 0    # validation questions may contain unseen word

        # soft label answers
        if self.train:
            a = np.zeros(self.n_answers, dtype=np.float32)
            for w, c in self.vqa[idx]['answers_w_scores']:
                try:
                    a[self.a_wtoi[w]] = c
                except:
                    continue
            a = np.asarray(a).reshape(-1)
        else:
            # return 0's for unknown test set answers
            a = 0

        # votes
        if self.train:
            n_votes = np.zeros(self.n_answers, dtype=np.float32)
            for w, c in self.vqa[idx]['answers']:
                try:
                    n_votes[self.a_wtoi[w]] = c
                except:
                    continue
            n_votes = np.asarray(n_votes).reshape(-1)
        else:
            # return 0's for unknown test set answers
            n_votes = 0

        # id of the question
        qid = self.vqa[idx]['question_id']

        # image sample
        iid = self.vqa[idx]['image_id']
        img = self.i_feat[str(iid)]
        bboxes = np.asarray(self.bbox[str(iid)])
        imsize = self.sizes[str(iid)]

        if np.logical_not(np.isfinite(img)).sum() > 0:
            raise ValueError

        # k sample
        k = 36

        # scale bounding boxes by image dimensions
        for i in range(k):
            bbox = bboxes[i]
            bbox[0] /= imsize[0]
            bbox[1] /= imsize[1]
            bbox[2] /= imsize[0]
            bbox[3] /= imsize[1]
            bboxes[i] = bbox

        # format
        q = np.asarray(q)
        qid = np.asarray(qid).reshape(-1)
        i = np.concatenate([img, bboxes], axis=1)
        k = np.asarray(k).reshape(1)

        return q, a, n_votes, qid, i, k, qlen
