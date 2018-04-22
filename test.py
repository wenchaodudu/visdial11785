import sys
import json
import pdb
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data_loader import get_loader
from model import Encoder


def main(config):
    print(config)

    eval_batch_size = 10
    dev_loader = get_loader('./data/visdial_data.h5', './data/data_img.h5', train=False, shuffle=False, batch_size=eval_batch_size)

    visdial = torch.load('model.pt')
    visdial.flatten_parameters()
    visdial.eval()

    best_rat1 = 0
    count = 0
    rat1 = 0
    rat2 = 0
    rat3 = 0
    rat5 = 0
    mrr = 0
    for i, data in enumerate(dev_loader):
        img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens = data
        prediction = visdial.evaluate(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens)
        prediction = prediction.data.cpu().numpy()
        labels = np.asarray(labels)
        for x in range(prediction.shape[0] // 10):
            ranks = prediction[x*10:(x+1)*10].argsort().argsort()
            rank = ranks[np.where(labels[x*10:(x+1)*10]==1)][0]
            if rank >= 9:
                rat1 += 1
            if rank >= 8:
                rat2 += 1
            if rank >= 7:
                rat3 += 1
            if rank >= 5:
                rat5 += 1
            mrr += 1 / (10 - rank)
            count += 1
    print(np.array([mrr, rat1, rat2, rat3, rat5]) / count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_saved', type=bool, default=False)
    parser.add_argument('--print_every_n_batches', type=int, default=10)
    parser.add_argument('--kl_weight', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--start_kl_weight', type=float, default=0)
    config = parser.parse_args()
    main(config)
