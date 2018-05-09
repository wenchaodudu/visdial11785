import json
import torch
import torch.utils.data as data
import h5py
import numpy as np
import pdb
import os
import _pickle as pickle


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_file, img_file, train=True):
        data = h5py.File(data_file, 'r')
        images = h5py.File(img_file, 'r')
        if train:
            self.captions = data['cap_train'][()]
            self.questions =  data['ques_train'][()]
            self.ques_len = data['ques_length_train'][()]
            self.ans_len = data['ans_length_train'][()]
            raw_opt_len = data['opt_length_train'][()]
            self.answers = data['ans_train'][()]
            raw_options = data['opt_train'][()]
            opt_list = data['opt_list_train'][()]
            self.images = images['images_train'][()]
            self.ans_idx = data['ans_index_train'][()]
        else:
            self.captions = data['cap_val'][()]
            self.questions =  data['ques_val'][()]
            self.ques_len = data['ques_length_val'][()]
            self.ans_len = data['ans_length_val'][()]
            raw_opt_len = data['opt_length_val'][()]
            self.answers = data['ans_val'][()]
            raw_options = data['opt_val'][()]
            opt_list = data['opt_list_val'][()]
            self.images = images['images_val'][()]
            self.ans_idx = data['ans_index_val'][()]

        self.length = self.captions.shape[0]
        self.options = np.zeros((self.length, 10, 100, 20), dtype=np.int32)
        self.opt_len = np.zeros((self.length, 10, 100), dtype=np.int32)
        for x in range(self.length):
            for y in range(10):
                self.options[x, y, :, :] = opt_list[raw_options[x][y]-1]
                self.opt_len[x, y, :] = raw_opt_len[raw_options[x][y]-1]

    def __getitem__(self, index):
        return self.images[index], self.captions[index], self.questions[index], \
               self.answers[index], self.options[index], self.ans_idx[index], \
               self.ques_len[index], self.ans_len[index], self.opt_len[index]

    def __len__(self):
        return self.length


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # seperate source and target sequences
    img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    '''
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    return src_seqs, src_lengths, trg_seqs, trg_lengths
    '''

    return img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens


def get_loader(h5_path, img_file, train=True, shuffle=True, batch_size=100):
    # build a custom dataset
    if train:
        dataset = Dataset(h5_path, img_file, train)
    else:
        dataset = Dataset(h5_path, img_file, train)

    # data loader for custome dataset
    # please see collate_fn for details
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=collate_fn)

    return data_loader

if __name__ == '__main__':
    get_loader('./data/visdial_data.h5', './data/data_img.h5', train=True, shuffle=True)

