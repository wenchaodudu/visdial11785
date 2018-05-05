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
    parser.add_argument('--batch_size', default=40, help='batch size')
    parser.add_argument('--lr', default=0.001, help='initial learning rate')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--model_path', default='./gen_', help='folder to output model checkpoints')
    parser.add_argument('--num_neg', default=100 , help='number of negative examples during training')
    parser.add_argument('--use_saved', action='store_true', help='use saved parameters for training')
    parser.add_argument('--baseline', action='store_true', help='baseline model')

    opt = parser.parse_args()
    print(opt)
    
    devloader = get_loader(opt.data_file, opt.img_file, train=False, batch_size=20)

    if opt.use_saved:
        net = torch.load(opt.model_path + 'torch_model.pt')
    else:
        raise NotImplementedError
    if opt.cuda:
        net.cuda()

    translation = json.load(open('data/visdial_params.json'))['ind2word']
    net.eval()
    for i, data in enumerate(devloader):
        img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, cap_lens, ques_lens, ans_lens, opt_lens = data
        opt_score = net.evaluate(img_seqs, cap_seqs, ques_seqs, opt_seqs, ques_lens, opt_lens)
        # batch_size * max_len * input_size
        text = opt_score[:, 0, :].max(dim=1)[1].data.cpu().numpy()
        try:
            eos = np.where(text == end_ind)[0][0]
        except:
            eos = len(text)
        res = [translation[text[i]] for i in range(eos)]
        res = ''.join(res)
        print(res)
    print('Finished Training')
