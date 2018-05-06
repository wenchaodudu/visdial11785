import pdb
import argparse
import torch
import time
import json
import sys
import numpy as np
from g_data_loader import *
from model import *

start_ind = 8834
end_ind = 8835

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_file', default='data/visdial_data.h5', help='path to visdial data hdf5 file')
    parser.add_argument('--img_file', default='data/data_img.h5', help='path to image hdf5 file')
    parser.add_argument('--training_epoch', default=20, help='training epoch')
    parser.add_argument('--batch_size', default=2, help='batch size')
    parser.add_argument('--lr', default=0.001, help='initial learning rate')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--model_path', default='', help='folder to output model checkpoints')
    parser.add_argument('--num_neg', default=100 , help='number of negative examples during training')
    parser.add_argument('--use_saved', action='store_true', help='use saved parameters for training')
    parser.add_argument('--baseline', action='store_true', help='baseline model')

    opt = parser.parse_args()
    print(opt)
    
    devloader = get_loader(opt.data_file, opt.img_file, train=False, batch_size=opt.batch_size)

    if opt.use_saved:
        if opt.cuda:
            net = torch.load(opt.model_path + 'gen_torch_model.pt')
        else:
            net = torch.load(opt.model_path + 'gen_torch_model.pt', map_location=lambda storage, loc: storage)
    else:
        raise NotImplementedError
    if opt.cuda:
        net.cuda()

    translation = json.load(open('data/visdial_params.json'))['ind2word']
    translation = np.asarray([translation[str(x)] for x in range(1, 8834)])
    net.eval()
    mrr, rat1, rat2, rat3, rat5 = 0, 0, 0, 0, 0
    count = 0
    for i, data in enumerate(devloader):
        if i % 100 == 1:
            print(i, mrr / count, rat1 / count)
        img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, cap_lens, ques_lens, ans_lens, opt_lens = data
        #output, opt_score = net.evaluate(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, opt.num_neg, 0.8)
        '''
        output = net.generate(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens)
        text = output.max(2)[1].data.cpu().numpy()
        for x in range(opt.batch_size):
            eos = np.where(text[x] == end_ind)
            if len(eos[0]):
                texts = text[x, :eos[0][0]]
            texts = ' '.join(translation[texts].tolist())
            print(texts)
        '''

        loss = net.evaluate(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens)
        prediction = loss.data.cpu().numpy()
        ranks = prediction.argsort(axis=1).argsort(axis=1) + 1
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
