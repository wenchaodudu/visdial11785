import pdb
import argparse
import torch
import torch.optim as optim
import time
import json
import sys
import numpy as np
from data_loader import *
from model import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_file', default='data/visdial_data.h5', help='path to visdial data hdf5 file')
    parser.add_argument('--img_file', default='data/data_img.h5', help='path to image hdf5 file')
    parser.add_argument('--training_epoch', default=20, help='training epoch')
    parser.add_argument('--batch_size', default=50, help='batch size')
    parser.add_argument('--lr', default=0.001, help='initial learning rate')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--model_path', default='', help='folder to output model checkpoints')
    parser.add_argument('--num_neg', default=100 , help='number of negative examples during training')
    parser.add_argument('--use_saved', action='store_true', help='use saved parameters for training')
    parser.add_argument('--baseline', action='store_true', help='baseline model')

    opt = parser.parse_args()
    print(opt)
    
    trainloader = get_loader(opt.data_file, opt.img_file, train=True, batch_size=opt.batch_size)
    devloader = get_loader(opt.data_file, opt.img_file, train=False, batch_size=50)

    embedding_dim = 200
    hidden_size = 200
    dictionary = json.load(open('data/visdial_params.json'))['word2ind']
    vocab_size = len(dictionary)+1
    word_vectors = np.random.uniform(low=-0.1, high=0.1, size=(vocab_size, embedding_dim))
    glove = open('data/glove.6B.200d.txt').readlines()
    found = 0
    for line in glove:
        word, vec = line.split(' ', 1)
        if word in dictionary:
            word_vectors[dictionary[word]] = np.fromstring(vec, sep=' ')
            found += 1
    print(found)
    
    if opt.use_saved:
        net = torch.load(opt.model_path + 'torch_model_3.pt')
        optimizer = torch.load(opt.model_path + 'optimizer.pt')
    else:
        if opt.baseline:
            net = Baseline(embedding_dim, hidden_size, vocab_size, word_vectors)
        else:
            net = Encoder(embedding_dim, hidden_size, vocab_size, word_vectors)
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr)
    if opt.cuda:
        net.cuda()

    best_res = 0
    for epoch in range(5, opt.training_epoch):
        # Train
        train_loss = 0
        net.train()
        last = time.time()
        for i, data in enumerate(trainloader):
            img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens = data
            optimizer.zero_grad()
            loss = net.loss(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, opt.num_neg)
            loss.backward()
            nn.utils.clip_grad_norm(net.parameters(), 1, 2)
            optimizer.step()
            train_loss += loss.cpu().data[0]
            if i % 100 == 0:
                print('Training loss: ', train_loss / min(i+1, 100), time.time() - last)
                train_loss = 0

        mrr, rat1, rat2, rat3, rat5 = 0, 0, 0, 0, 0
        count = 0
        for i, data in enumerate(devloader):
            img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens = data
            #ans_score, opt_score = net.evaluate(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens)
            opt_score = net.evaluate(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens)
            prediction = opt_score.data.cpu().numpy()
            ranks = 100 - prediction.argsort(axis=1).argsort(axis=1)
            truth = np.concatenate(ans_idx_seqs) - 1
            truth_rank = ranks[np.arange(ranks.shape[0]), truth]
            rat1 += np.sum(truth_rank <= 1)
            rat2 += np.sum(truth_rank <= 2)
            rat3 += np.sum(truth_rank <= 3)
            rat5 += np.sum(truth_rank <= 5)
            mrr += np.sum(1 / truth_rank)
            count += ranks.shape[0]

        results = np.asarray([mrr, rat1, rat2, rat3, rat5]) / count
        print(results)
        if results[0] > best_res:
            best_res = results[0]
            torch.save(net, opt.model_path + 'torch_model_best.pt')
        torch.save(optimizer, opt.model_path + 'optimizer.pt')

        print('Saving model...')
        torch.save(net, opt.model_path + 'torch_model_' + str(epoch) + '.pt')

        print('Learning rate: ', opt.lr)
        if epoch % 20 == 19:
            opt.lr *= 0.5
            optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    print('Finished Training')
