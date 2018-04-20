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
    parser.add_argument('--training_epoch', default=10, help='training epoch')
    parser.add_argument('--batch_size', default=128, help='batch size')
    parser.add_argument('--lr', default=0.001, help='initial learning rate')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--model_path', default='', help='folder to output model checkpoints')
    parser.add_argument('--num_neg', default=25 , help='number of negative examples during training')

    opt = parser.parse_args()
    print(opt)
    
    trainloader = get_loader(opt.data_file, opt.img_file, train=True, batch_size=opt.batch_size)
    devloader = get_loader(opt.data_file, opt.img_file, train=False, batch_size=10)

    dictionary = json.load(open('data/visdial_params.json'))['word2ind']
    word_vectors = np.random.uniform(low=-0.1, high=0.1, size=(len(dictionary)+1, 200))
    glove = open('data/glove.6B.200d.txt').readlines()
    found = 0
    for line in glove:
        word, vec = line.split(' ', 1)
        if word in dictionary:
            word_vectors[dictionary[word]] = np.fromstring(vec, sep=' ')
            found += 1
    print(found)
    
    net = Encoder(200, 200, 8834, word_vectors)
    if opt.cuda:
        net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

    best_res = 0
    for epoch in range(opt.training_epoch):
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
            train_loss += loss.cpu().data[0] / opt.batch_size
            if i % 10 == 0:
                print('Training loss: ', train_loss / min(i+1, 10), time.time() - last)
                train_loss = 0

        mrr, rat1, rat2, rat3, rat5 = 0, 0, 0, 0, 0
        count = 0
        for i, data in enumerate(devloader):
            img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens = data
            ans_score, opt_score = net.evaluate(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens)
            prediction = opt_score.data.cpu().numpy()
            ranks = 100 - prediction.argsort(axis=1).argsort(axis=1)
            truth = np.concatenate(ans_idx_seqs)
            truth_rank = ranks[np.arange(ranks.shape[0]), truth]
            rat1 += np.sum(truth_rank <= 1)
            rat2 += np.sum(truth_rank <= 2)
            rat3 += np.sum(truth_rank <= 3)
            rat5 += np.sum(truth_rank <= 5)
            mrr += np.sum(1 / truth_rank)
            count += ranks.shape[0]

        results = np.asarray([mrr, rat1, rat2, rat3, rat5]) / count
        if results[0] > best_res:
            best_res = results[0]
            torch.save(net, opt.model_path + 'torch_model_best.pt')

        print('Saving model...')
        torch.save(net, opt.model_path + 'torch_model_' + str(epoch) + '.pt')

        print('Learning rate: ', opt.learning_rate)
        if epoch % 20 == 19:
            opt.lr *= 0.5
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    print('Finished Training')