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


class GRUEncoder(nn.Module):
    """ 
    input: (batch_size, seq_len, embedding_dim)
    output: (batch_size, hidden_size * direction)
    """
    def __init__(self, input_size, hidden_size):
        super(GRUEncoder, self).__init__()
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
        self.hidden_size = hidden_size
        self.embed = Embedding(vocab_size, input_size)
        self.uencoder = GRUEncoder(input_size, hidden_size)
        #self.fencoder = nn.Bilinear(4096, hidden_size, hidden_size)
        self.fencoder = nn.Linear(4096+hidden_size, hidden_size)
        self.hencoder = GRUEncoder(hidden_size, hidden_size)
        self.score = nn.Bilinear(hidden_size, hidden_size, 2)

    def embed_utterance(self, src_seqs, src_lengths):
        src_len, perm_idx = src_lengths.sort(0, descending=True)
        src_sortedseqs = self.embed(src_seqs[perm_idx])
        packed_input = pack_padded_sequence(src_sortedseqs, src_len.cpu().numpy(), batch_first=True)
        src_vec = self.uencoder(packed_input)
        return src_vec[perm_idx.sort(0, descending=True)[1]]

    def encode_feature(self, img, utt):
        #return self.fencoder(img, utt)
        return F.relu(self.fencoder(torch.stack((img, utt), dim=1)))

    def forward(self, img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens):
        img_seqs = Variable(torch.from_numpy(np.vstack(img_seqs)))

        ques_seqs = torch.from_numpy(np.concatenate(ques_seqs).astype(np.int32)).long()
        ans_seqs = torch.from_numpy(np.concatenate(ans_seqs).astype(np.int32)).long()
        opt_seqs = torch.from_numpy(np.concatenate(opt_seqs).astype(np.int32)).long()
        opt_seqs = opt_seqs.view(-1, opt_seqs.size(2))

        ques_lens = torch.from_numpy(np.concatenate(ques_lens).astype(np.int32)).long()
        ans_lens = torch.from_numpy(np.concatenate(ans_lens).astype(np.int32)).long()
        opt_lens = torch.from_numpy(np.concatenate(opt_lens).astype(np.int32)).long()
        opt_lens = opt_lens.view(-1)
        
        ques_vec = self.embed_utterance(ques_seqs, ques_lens)
        ans_vec = self.embed_utterance(ans_seqs, ans_lens)
        opt_vec = self.embed_utterance(opt_seqs, opt_lens)

        ques_feat = self.encode_feature(img_seqs.repeat(1, 10).view(-1, 4096), ques_vec)
        ans_feat = self.encode_feature(img_seqs.repeat(1, 10).view(-1, 4096), ans_vec)
        opt_feat = self.encode_feature(img_seqs.repeat(1, 1000).view(-1, 4096), opt_vec)
        ans_logits = self.score(ques_feat, ans_feat)
        opt_logits = self.score(ques_feat.repeat(1, 100).view(-1, self.hidden_size), opt_feat)
        ans_score = F.log_softmax(ans_logits, dim=1)
        opt_score = F.log_softmax(opt_logits, dim=1)

        return -(ans_score[:, 1].sum() * 100 + opt_score[:, 0].sum()) / (ans_score.size(0) * 100 + opt_score.size(0))

