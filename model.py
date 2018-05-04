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
        

class Baseline(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, word_vectors):
        super(Baseline, self).__init__()
        self.hidden_size = hidden_size
        self.embed = Embedding(vocab_size, input_size, word_vectors, trainable=False)
        self.qencoder = GRUEncoder(input_size, hidden_size)
        self.aencoder = GRUEncoder(input_size, hidden_size)
        self.score = nn.Bilinear(4096+hidden_size, hidden_size, 1)
        self.criterion = nn.CrossEntropyLoss()

    def embed_utterance(self, src_seqs, src_lengths, encoder):
        src_len, perm_idx = src_lengths.sort(0, descending=True)
        src_sortedseqs = self.embed(Variable(src_seqs[perm_idx]))
        packed_input = pack_padded_sequence(src_sortedseqs, src_len.cpu().numpy(), batch_first=True)
        src_vec = encoder(packed_input)
        return src_vec[perm_idx.sort(0)[1]]

    def forward(self, img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, num_neg):
        img_seqs = Variable(torch.from_numpy(np.vstack(img_seqs))).cuda()

        ques_seqs = torch.from_numpy(np.concatenate(ques_seqs).astype(np.int32)).long().cuda()
        ans_seqs = torch.from_numpy(np.concatenate(ans_seqs).astype(np.int32)).long().cuda()
        opt_seqs = torch.from_numpy(np.concatenate(opt_seqs).astype(np.int32)).long().cuda()
        if num_neg < 100:
            rand_ind = np.random.choice(100, num_neg, False)
            rand_ind = torch.from_numpy(rand_ind).long().cuda()
            opt_seqs = opt_seqs[:, rand_ind, :]
        opt_seqs = opt_seqs.view(-1, opt_seqs.size(2))

        ques_lens = torch.from_numpy(np.concatenate(ques_lens).astype(np.int32)).long().cuda()
        ans_lens = torch.from_numpy(np.concatenate(ans_lens).astype(np.int32)).long().cuda()
        opt_lens = torch.from_numpy(np.concatenate(opt_lens).astype(np.int32)).long().cuda()
        if num_neg < 100:
            opt_lens = opt_lens[:, rand_ind]
        opt_lens = opt_lens.view(-1)
        
        ques_vec = self.embed_utterance(ques_seqs, ques_lens, self.qencoder)
        ans_vec = self.embed_utterance(ans_seqs, ans_lens, self.aencoder)
        opt_vec = self.embed_utterance(opt_seqs, opt_lens, self.aencoder)

        feature = torch.cat((img_seqs.repeat(1, 10).view(-1, 4096), ques_vec), dim=1)
        feature = feature.repeat(1, num_neg).view(-1, 4096+self.hidden_size)
        logits = self.score(feature, opt_vec)

        return logits

    def loss(self, img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, num_neg):
        opt_logits = self.forward(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, num_neg)
        opt_logits = opt_logits.view(-1, num_neg)
        ans_idx_seqs = torch.from_numpy(np.concatenate(ans_idx_seqs).astype(np.int32)).long().cuda()
        
        return self.criterion(opt_logits, Variable(ans_idx_seqs-1))

    def evaluate(self, img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens):
        opt_logits = self.forward(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, 100)
        opt_logits = opt_logits.view(-1, 100)

        return opt_logits
 

class SimpleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, word_vectors):
        super(SimpleEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed = Embedding(vocab_size, input_size, word_vectors, trainable=False)
        self.qencoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.aencoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.score = nn.Bilinear(hidden_size, hidden_size, 1)
        self.criterion = nn.CrossEntropyLoss()

    def embed_utterance(self, src_seqs, src_lengths, encoder):
        src_len, perm_idx = src_lengths.sort(0, descending=True)
        src_sortedseqs = self.embed(Variable(src_seqs[perm_idx]))
        packed_input = pack_padded_sequence(src_sortedseqs, src_len.cpu().numpy(), batch_first=True)
        pdb.set_trace()
        output, hidden = encoder(packed_input)
        src_vec = pad_packed_sequence(hidden, batch_first=True)
        return src_vec[perm_idx.sort(0)[1]]

    def forward(self, img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, num_neg):
        img_seqs = Variable(torch.from_numpy(np.vstack(img_seqs))).cuda()

        ques_seqs = torch.from_numpy(np.concatenate(ques_seqs).astype(np.int32)).long().cuda()
        ans_seqs = torch.from_numpy(np.concatenate(ans_seqs).astype(np.int32)).long().cuda()
        opt_seqs = torch.from_numpy(np.concatenate(opt_seqs).astype(np.int32)).long().cuda()
        if num_neg < 100:
            rand_ind = np.random.choice(100, num_neg, False)
            rand_ind = torch.from_numpy(rand_ind).long().cuda()
            opt_seqs = opt_seqs[:, rand_ind, :]
        opt_seqs = opt_seqs.view(-1, opt_seqs.size(2))

        ques_lens = torch.from_numpy(np.concatenate(ques_lens).astype(np.int32)).long().cuda()
        ans_lens = torch.from_numpy(np.concatenate(ans_lens).astype(np.int32)).long().cuda()
        opt_lens = torch.from_numpy(np.concatenate(opt_lens).astype(np.int32)).long().cuda()
        if num_neg < 100:
            opt_lens = opt_lens[:, rand_ind]
        opt_lens = opt_lens.view(-1)
        
        ques_vec = self.embed_utterance(ques_seqs, ques_lens, self.qencoder)
        ans_vec = self.embed_utterance(ans_seqs, ans_lens, self.aencoder)
        opt_vec = self.embed_utterance(opt_seqs, opt_lens, self.aencoder)

        '''
        ques_feat = self.encode_feature(img_seqs.repeat(1, 10).view(-1, 4096), ques_vec)
        ans_feat = self.encode_feature(img_seqs.repeat(1, 10).view(-1, 4096), ans_vec)
        opt_feat = self.encode_feature(img_seqs.repeat(1, 10*num_neg).view(-1, 4096), opt_vec)
        ans_logits = self.score(ques_feat, ans_feat)
        opt_logits = self.score(ques_feat.repeat(1, num_neg).view(-1, self.hidden_size), opt_feat)
        '''

        return ans_logits, opt_logits

    def loss(self, img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, num_neg):
        ans_logits, opt_logits = self.forward(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, num_neg)
        #ans_logits = ans_logits.view(-1, 10)
        opt_logits = opt_logits.view(-1, 100)
        ans_idx_seqs = torch.from_numpy(np.concatenate(ans_idx_seqs).astype(np.int32)).long().cuda()
        '''
        ans_score = F.log_softmax(ans_logits, dim=1)
        opt_score = F.log_softmax(opt_logits, dim=1)

        return -(ans_score[:, 1].sum() * num_neg + opt_score[:, 0].sum()) / (ans_score.size(0) * num_neg + opt_score.size(0))
        '''
        return self.criterion(opt_logits, Variable(ans_idx_seqs-1))

    def evaluate(self, img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens):
        ans_logits, opt_logits = self.forward(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, 100)
        '''
        ans_score = F.softmax(ans_logits, dim=1)
        opt_score = F.softmax(opt_logits, dim=1)

        return ans_score.view(-1, 10, 2)[:, :, 1], opt_score.view(-1, 100, 2)[:, :, 1]
        '''
        opt_logits = opt_logits.view(-1, 100)

        return opt_logits
 
class MatchingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, word_vectors):
        super(MatchingNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = input_size
        self.embed = Embedding(vocab_size, input_size, word_vectors, trainable=False)
        self.uencoder = nn.GRU(input_size, hidden_size)
        #self.fencoder = nn.Bilinear(4096, hidden_size, 1)
        self.hencoder = GRUEncoder(hidden_size, hidden_size)
        self.score = nn.Bilinear(hidden_size, hidden_size, 1)
        self.criterion = nn.CrossEntropyLoss()
        self.ww_M = nn.Linear(hidden_size, hidden_size)
        self.iw_M = nn.Linear(128, input_size)
        self.iu_M = nn.Linear(128, hidden_size)
        self.ww_conv = nn.Conv2d(2, 8, kernel_size=(3, 3), padding=1)
        self.iw_conv = nn.Conv2d(2, 8, kernel_size=(3, 3), padding=1)
        self.ww_pool = nn.MaxPool2d((3, 3), stride=(3, 3))
        self.iw_pool = nn.MaxPool2d((3, 3), stride=(3, 3))
        self.fc1 = nn.Linear(288 + 480, 100)
        self.fc2 = nn.Linear(100, 1)

    def pad_seq(self, seqs):
        size, length, dim = list(seqs.size())
        if length < 20:
            seqs = torch.cat((seqs, Variable(torch.zeros(size, 20 - length, dim).float().cuda())), dim=1)
        return seqs

    def embed_utterance(self, src_seqs, src_lengths):
        src_len, perm_idx = src_lengths.sort(0, descending=True)
        src_sortedseqs = self.embed(Variable(src_seqs[perm_idx]))
        packed_input = pack_padded_sequence(src_sortedseqs, src_len.cpu().numpy(), batch_first=True)
        src_output, _ = self.uencoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        src_sortedseqs = self.pad_seq(src_sortedseqs)
        src_hidden = self.pad_seq(src_hidden)
        rev_idx = perm_idx.sort(0)[1]
        return src_hidden[rev_idx], src_sortedseqs[rev_idx]

    def encode_feature(self, img, utt):
        #return self.fencoder(img, utt)
        return F.relu(self.fencoder(torch.cat((img, utt), dim=1)))

    def forward(self, img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, num_neg):
        img_seqs = Variable(torch.from_numpy(np.vstack(img_seqs))).cuda()
        batch_size = img_seqs.size(0)

        ques_seqs = torch.from_numpy(np.concatenate(ques_seqs).astype(np.int32)).long().cuda()
        ans_seqs = torch.from_numpy(np.concatenate(ans_seqs).astype(np.int32)).long().cuda()
        opt_seqs = torch.from_numpy(np.concatenate(opt_seqs).astype(np.int32)).long().cuda()
        if num_neg < 100:
            rand_ind = np.random.choice(100, num_neg, False)
            rand_ind = torch.from_numpy(rand_ind).long().cuda()
            opt_seqs = opt_seqs[:, rand_ind, :]
        opt_seqs = opt_seqs.view(-1, opt_seqs.size(2))

        ques_lens = torch.from_numpy(np.concatenate(ques_lens).astype(np.int32)).long().cuda()
        ans_lens = torch.from_numpy(np.concatenate(ans_lens).astype(np.int32)).long().cuda()
        opt_lens = torch.from_numpy(np.concatenate(opt_lens).astype(np.int32)).long().cuda()
        if num_neg < 100:
            opt_lens = opt_lens[:, rand_ind]
        opt_lens = opt_lens.view(-1)
        
        ques_hidden, ques_embed = self.embed_utterance(ques_seqs, ques_lens)
        ans_hidden, ans_embed = self.embed_utterance(ans_seqs, ans_lens)
        opt_hidden, opt_embed = self.embed_utterance(opt_seqs, opt_lens)

        # question answer matching
        ques_hidden = ques_hidden.unsqueeze(1).expand(batch_size * 10, num_neg, 20, self.hidden_size).contiguous().view(-1, 20, self.hidden_size)
        ques_embed = ques_embed.unsqueeze(1).expand(batch_size * 10, num_neg, 20, self.embedding_dim).contiguous().view(-1, 20, self.embedding_dim)
        utt_sim = self.ww_M(ques_hidden)
        utt_sim = torch.bmm(utt_sim, opt_hidden.transpose(1, 2))
        word_sim = torch.bmm(ques_embed, opt_embed.transpose(1, 2))
        ww_conv_input = torch.cat((word_sim.unsqueeze(1), utt_sim.unsqueeze(1)), dim=1)
        ww_conv_output = F.relu(self.ww_conv(ww_conv_input))
        ww_conv_output = self.ww_pool(ww_conv_output)
        ww_conv_output = ww_conv_output.view(ww_conv_output.size(0), -1)

        # image answer matching
        img_seqs = img_seqs.unsqueeze(1).expand(batch_size, 10 * num_neg, 4096).contiguous().view(-1, 4096).view(-1, 32, 128)
        iw_sim = self.iw_M(img_seqs)
        iw_sim = torch.bmm(iw_sim, opt_embed.transpose(1, 2))
        iu_sim = self.iu_M(img_seqs)
        iu_sim = torch.bmm(iu_sim, opt_hidden.transpose(1, 2))
        iw_conv_input = torch.cat((iw_sim.unsqueeze(1), iu_sim.unsqueeze(1)), dim=1)
        iw_conv_output = F.relu(self.iw_conv(iw_conv_input))
        iw_conv_output = self.iw_pool(iw_conv_output)
        iw_conv_output = iw_conv_output.view(iw_conv_output.size(0), -1)

        logits = self.fc2(F.relu(self.fc1(torch.cat((ww_conv_output, iw_conv_output), dim=1))))

        return logits

    def loss(self, img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, num_neg):
        logits = self.forward(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, num_neg)
        #ans_logits = ans_logits.view(-1, 10)
        opt_logits = logits.view(-1, 100)
        ans_idx_seqs = torch.from_numpy(np.concatenate(ans_idx_seqs).astype(np.int32)).long().cuda()
        '''
        ans_score = F.log_softmax(ans_logits, dim=1)
        opt_score = F.log_softmax(opt_logits, dim=1)

        return -(ans_score[:, 1].sum() * num_neg + opt_score[:, 0].sum()) / (ans_score.size(0) * num_neg + opt_score.size(0))
        '''
        return self.criterion(opt_logits, Variable(ans_idx_seqs-1))

    def evaluate(self, img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens):
        logits = self.forward(img_seqs, cap_seqs, ques_seqs, ans_seqs, opt_seqs, ans_idx_seqs, ques_lens, ans_lens, opt_lens, 100)
        '''
        ans_score = F.softmax(ans_logits, dim=1)
        opt_score = F.softmax(opt_logits, dim=1)

        return ans_score.view(-1, 10, 2)[:, :, 1], opt_score.view(-1, 100, 2)[:, :, 1]
        '''
        opt_logits = opt_logits.view(-1, 100)

        return opt_logits
               
