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
import os
import json
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.optim.lr_scheduler import MultiStepLR

from sparse_graph_model import Model
from torch_dataset import *
from utils import *

def eval_model(args):

    """
        Computes the VQA accuracy over the validation set
        using a pre-trained model
    """

    # Check that the model path is accurate
    if args.model_path and os.path.isfile(args.model_path):
        print('Resuming from checkpoint %s' % (args.model_path))
    else:
        raise SystemExit('Need to provide model path.')

    # Set random seed
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, script requires cuda')

    # Load the validation set
    print('Loading data')
    dataset = VQA_Dataset(args.data_dir, args.emb, train=False)
    loader = DataLoader(dataset, batch_size=args.bsize,
                        shuffle=False, num_workers=5, 
                        collate_fn=collate_fn)

    # Print data and model parameters
    print('Parameters:\n\t'
          'vocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d' 
          '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words, args.emb,
                                                    dataset.feat_dim,
                                                    args.hid,
                                                    dataset.n_answers))
    # Define the model
    model = Model(vocab_size=dataset.q_words,
                  emb_dim=args.emb,
                  feat_dim=dataset.feat_dim,
                  hid_dim=args.hid,
                  out_dim=dataset.n_answers,
                  dropout=args.dropout,
                  pretrained_wemb=dataset.pretrained_wemb,
                  neighbourhood_size=args.neighbourhood_size)

    # move to CUDA
    model = model.cuda()

    # Restore pre-trained model
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['state_dict'])
    model.train(False)

    # Compute accuracy
    result = []
    correct = 0
    for step, next_batch in tqdm(enumerate(loader)):
        # move batch to cuda
        q_batch, _, vote_batch, i_batch, k_batch, qlen_batch = \
            batch_to_cuda(next_batch, volatile=True)

        # get predictions
        output, _ = model(q_batch, i_batch, k_batch, qlen_batch)
        qid_batch = next_batch[3]
        _, oix = output.data.max(1)
        # record predictions
        for i, qid in enumerate(qid_batch):
            result.append({
                'question_id': int(qid.numpy()),
                'answer': dataset.a_itow[oix[i]]
            })
        # compute batch accuracy
        correct += total_vqa_score(output, vote_batch)

    # compute and print average accuracy
    acc = correct/dataset.n_questions*100
    print("accuracy: {} %".format(acc))

    # save predictions
    json.dump(result, open('result.json', 'w'))
    print('Validation done')

def train(args):

    """
        Train a VQA model using the training set
    """

    # set random seed
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, script requires cuda')

    # Load the VQA training set
    print('Loading data')
    dataset = VQA_Dataset(args.data_dir, args.emb)
    loader = DataLoader(dataset, batch_size=args.bsize,
                        shuffle=True, num_workers=5, collate_fn=collate_fn)

    # Load the VQA validation set
    dataset_test = VQA_Dataset(args.data_dir, args.emb, train=False)
    test_sampler = RandomSampler(dataset_test)
    loader_test = iter(DataLoader(dataset_test,
                                  batch_size=args.bsize,
                                  sampler=test_sampler,
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=collate_fn))

    n_batches = len(dataset)//args.bsize

    # Print data and model parameters
    print('Parameters:\n\t'
          'vocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d'
          '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words, args.emb,
                                                    dataset.feat_dim,
                                                    args.hid,
                                                    dataset.n_answers))
    print('Initializing model')

    model = Model(vocab_size=dataset.q_words,
                  emb_dim=args.emb,
                  feat_dim=dataset.feat_dim,
                  hid_dim=args.hid,
                  out_dim=dataset.n_answers,
                  dropout=args.dropout,
                  neighbourhood_size=args.neighbourhood_size,
                  pretrained_wemb=dataset.pretrained_wemb)

    criterion = nn.MultiLabelSoftMarginLoss()

    # Move it to GPU
    model = model.cuda()
    criterion = criterion.cuda()

    # Define the optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Continue training from saved model
    start_ep = 0
    if args.model_path and os.path.isfile(args.model_path):
        print('Resuming from checkpoint %s' % (args.model_path))
        ckpt = torch.load(args.model_path)
        start_ep = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # Update the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

    # Learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=[30], gamma=0.5)
    scheduler.last_epoch = start_ep - 1

    # Train iterations
    print('Start training.')
    for ep in range(start_ep, start_ep+args.ep):

        scheduler.step()
        ep_loss = 0.0
        ep_correct = 0.0
        ave_loss = 0.0
        ave_correct = 0.0
        losses = []

        for step, next_batch in tqdm(enumerate(loader)):

            model.train()
            # Move batch to cuda
            q_batch, a_batch, vote_batch, i_batch, k_batch, qlen_batch = \
                batch_to_cuda(next_batch)

            # forward pass
            output, adjacency_matrix = model(
                q_batch, i_batch, k_batch, qlen_batch)

            loss = criterion(output, a_batch)

            # Compute batch accuracy based on vqa evaluation
            correct = total_vqa_score(output, vote_batch)
            ep_correct += correct
            ep_loss += loss.data[0]
            ave_correct += correct
            ave_loss += loss.data[0]
            losses.append(loss.cpu().data[0])

            # This is a 40 step average
            if step % 40 == 0 and step != 0:
                print('  Epoch %02d(%03d/%03d), ave loss: %.7f, ave accuracy: %.2f%%' %
                      (ep+1, step, n_batches, ave_loss/40,
                       ave_correct * 100 / (args.bsize*40)))

                ave_correct = 0
                ave_loss = 0

            # Compute gradient and do optimisation step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save model and compute validation accuracy every 400 steps
            if step % 400 == 0:
                epoch_loss = ep_loss / n_batches
                epoch_acc = ep_correct * 100 / (n_batches * args.bsize)

                save(model, optimizer, ep, epoch_loss, epoch_acc,
                     dir=args.save_dir, name=args.name+'_'+str(ep+1))

                # compute validation accuracy over a small subset of the validation set
                test_correct = 0
                model.train(False)

                for i in range(10):
                    test_batch = next(loader_test)
                    q_batch, a_batch, vote_batch, i_batch, k_batch, qlen_batch = \
                        batch_to_cuda(test_batch, volatile=True)
                    output, _ = model(q_batch, i_batch, k_batch, qlen_batch)
                    test_correct += total_vqa_score(output, vote_batch)

                model.train(True)
                acc = test_correct/(10*args.bsize)*100
                print("Validation accuracy: {:.2f} %".format(acc))

        # save model and compute accuracy for epoch
        epoch_loss = ep_loss / n_batches
        epoch_acc = ep_correct * 100 / (n_batches * args.bsize)

        save(model, optimizer, ep, epoch_loss, epoch_acc,
             dir=args.save_dir, name=args.name+'_'+str(ep+1))

        print('Epoch %02d done, average loss: %.3f, average accuracy: %.2f%%' % (
              ep+1, epoch_loss, epoch_acc))

def test(args):

    """
        Creates a result.json for predictions on
        the test set
    """
    # Check that the model path is accurate
    if args.model_path and os.path.isfile(args.model_path):
        print('Resuming from checkpoint %s' % (args.model_path))
    else:
        raise SystemExit('Need to provide model path.')

    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, script requires CUDA')
 
    print('Loading data')
    dataset = VQA_Dataset_Test(args.data_dir, args.emb, train=False)
    loader = DataLoader(dataset, batch_size=args.bsize, 
                        shuffle=False, num_workers=5, 
                        collate_fn=collate_fn)

    # Print data and model parameters
    print('Parameters:\n\t'
          'vocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d' 
          '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words, args.emb,
                                                    dataset.feat_dim,
                                                    args.hid,
                                                    dataset.n_answers))

    # Define model
    model = Model(vocab_size=dataset.q_words,
                  emb_dim=args.emb,
                  feat_dim=dataset.feat_dim,
                  hid_dim=args.hid,
                  out_dim=dataset.n_answers,
                  dropout=args.dropout,
                  pretrained_wemb=dataset.pretrained_wemb,
                  neighbourhood_size=args.neighbourhood_size)
 
    # move to CUDA
    model = model.cuda()
 
    # Restore pre-trained model
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['state_dict'])
    model.train(False)

    result = []
    for step, next_batch in tqdm(enumerate(loader)):
        # Batch preparation
        q_batch, _, _, i_batch, k_batch, qlen_batch = \
            batch_to_cuda(next_batch, volatile=True)
 
        # get predictions
        output, _ = model(q_batch, i_batch, k_batch, qlen_batch)
        qid_batch = next_batch[3]
        _, oix = output.data.max(1)
        # record predictions
        for i, qid in enumerate(qid_batch):
            result.append({
                'question_id': int(qid.numpy()),
                'answer': dataset.a_itow[oix[i]]
            })
 
    json.dump(result, open('result.json', 'w'))
    print('Testing done')

def trainval(args):

    """
        Train a VQA model using the training + validation set
    """
 
    # set random seed
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, script requires CUDA.')
    
    # load train+val sets for training
    print ('Loading data')
    dataset = VQA_Dataset_Test(args.data_dir, args.emb)
    loader = DataLoader(dataset, batch_size=args.bsize, 
                        shuffle=True, num_workers=5, 
                        collate_fn=collate_fn)
    n_batches = len(dataset)//args.bsize

    # Print data and model parameters
    print ('Parameters:\n\tvocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d\
            \n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words, args.emb, dataset.feat_dim,
                args.hid, dataset.n_answers))
    print ('Initializing model')
 
    model = Model(vocab_size=dataset.q_words,
                  emb_dim=args.emb,
                  feat_dim=dataset.feat_dim,
                  hid_dim=args.hid,
                  out_dim=dataset.n_answers,
                  dropout=args.dropout,
                  neighbourhood_size=args.neighbourhood_size,
                  pretrained_wemb=dataset.pretrained_wemb)
 
    criterion = nn.MultiLabelSoftMarginLoss()

    # Move it to GPU
    model = model.cuda()
    criterion = criterion.cuda()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
 
    # Continue training from saved model
    start_ep = 0
    if args.model_path and os.path.isfile(args.model_path):
        print ('Resuming from checkpoint %s' % (args.model_path))
        ckpt = torch.load(args.model_path)
        start_ep = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
 
    # ensure you can load with new lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
 
    # learner rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=[30], gamma=0.5)
    scheduler.last_epoch = start_ep - 1
 
    # Training script
    print ('Start training.')
    for ep in range(start_ep, start_ep+args.ep):
        scheduler.step()
        ep_loss = 0.0
        ep_correct = 0.0
        ave_loss = 0.0
        ave_correct = 0.0
        losses = []
        for step, next_batch in tqdm(enumerate(loader)):
            model.train()
            # batch to gpu
            q_batch, a_batch, vote_batch, i_batch, k_batch, qlen_batch = \
                batch_to_cuda(next_batch)

             # Do model forward
            output, adjacency_matrix = model(
                q_batch, i_batch, k_batch, qlen_batch)
            
            loss = criterion(output, a_batch)
 
            # compute accuracy based on vqa evaluation
            correct = total_vqa_score(output, vote_batch)
            ep_correct += correct
            ep_loss += loss.data[0]
            ave_correct += correct
            ave_loss += loss.data[0]
            losses.append(loss.cpu().data[0])
            # This is a 40 step average
            if step % 40 == 0 and step != 0:
                print('  Epoch %02d(%03d/%03d), ave loss: %.7f, ave accuracy: %.2f%%' %
                      (ep+1, step, n_batches, ave_loss/40,
                       ave_correct * 100 / (args.bsize*40)))

                ave_correct = 0
                ave_loss = 0
                ave_correct = ave_loss = ave_sparsity = 0
 
            # compute gradient and do optim step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save model and compute accuracy for epoch
        epoch_loss = ep_loss / n_batches
        epoch_acc = ep_correct * 100 / (n_batches * args.bsize)


        save(model, optimizer, ep, epoch_loss, epoch_acc,
             dir=args.save_dir, name=args.name+'_'+str(ep+1))

        print('Epoch %02d done, average loss: %.3f, average accuracy: %.2f%%' % (
              ep+1, epoch_loss, epoch_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description='Conditional Graph Convolutions for VQA')
    parser.add_argument('--train', action='store_true',
                        help='set this to training mode.')
    parser.add_argument('--trainval', action='store_true',
                        help='set this to train+val mode.')
    parser.add_argument('--eval', action='store_true',
                        help='set this to evaluation mode.')
    parser.add_argument('--test', action='store_true',
                        help='set this to test mode.')
    parser.add_argument('--lr', metavar='', type=float,
                        default=1e-4, help='initial learning rate')
    parser.add_argument('--ep', metavar='', type=int,
                        default=40, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int,
                        default=64, help='batch size.')
    parser.add_argument('--hid', metavar='', type=int,
                        default=1024, help='hidden dimension')
    parser.add_argument('--emb', metavar='', type=int, default=300,
                        help='question embedding dimension')
    parser.add_argument('--neighbourhood_size', metavar='', type=int, default=16,
                        help='number of graph neighbours to consider')
    parser.add_argument('--data_dir', metavar='', type=str, default='./data',
                        help='path to data directory')
    parser.add_argument('--save_dir', metavar='', type=str, default='./save')
    parser.add_argument('--name', metavar='', type=str,
                        default='model', help='model name')
    parser.add_argument('--dropout', metavar='', type=float, default=0.5,
                        help='probability of dropping out FC nodes during training')
    parser.add_argument('--model_path', metavar='', type=str,
                        help='trained model path.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))
    if args.train:
        train(args)
    if args.trainval:
        trainval(args)
    if args.eval:
        eval_model(args)
    if args.test:
        test(args)
    if not args.train and not args.eval and not args.trainval and not args.test:
        parser.print_help()
