import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pdb 


class Embedding(nn.Module):
    """ 
    input: (batch_size, seq_length)
    output: (batch_size, seq_length, embedding_dim)
    """
    def __init__(self, num_embeddings, embedding_dim, init_embedding=None, trainable=False):
        # init_embedding: 2d matrix of pre-trained word embedding
        # row 0 used for padding
        super(Embedding, self).__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))

        self.padding_idx = 0 
        self.max_norm = None
        self.norm_type = 2 
        self.scale_grad_by_freq = False
        self.sparse = False

        self.reset_parameters(init_embedding)
        self.weight.requires_grad = trainable

    def reset_parameters(self, init_embedding=None):
        if not (init_embedding is None):
            self.weight.data.copy_(torch.from_numpy(init_embedding))
        else:
            self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, input):
        padding_idx = self.padding_idx
        if padding_idx is None:
            padding_idx = -1
        return self._backend.Embedding.apply(
            input, self.weight,
            padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse
        )

    def __repr__(self):
        return nn.Embedding.__repr__(self)


class UtteranceEncoder(nn.Module):
    """ 
    input: (batch_size, seq_len, embedding_dim)
    output: (batch_size, hidden_size * direction)
    """
    def __init__(self, input_size, hidden_size):
        super(UtteranceEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1 
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers,
                           bidirectional=False, batch_first=True)

    def forward(self, input):
        output, h = self.rnn(input)
        # return output
        return torch.transpose(h, 0, 1).contiguous().view(-1, self.hidden_size)

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        return h


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super(Encoder, self).__init__()
        self.embed = Embedding(vocab_size, input_size)

    def forward(self, img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens):
        pdb.set_trace()
        ques_seqs = torch.from_numpy(np.concatenate(ques_seqs))
        ans_seqs = torch.from_numpy(np.concatenate(ans_seqs))
        opt_seqs = torch.from_numpy(np.concatenate(opt_seqs))

