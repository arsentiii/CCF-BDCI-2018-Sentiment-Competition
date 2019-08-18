import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .init_util import *
import math

class HUARN(nn.Module):
    def __init__(self, input_size, hidden_size, aspect_embedding, word_embedding, pos_vocab_size, device, stage, aspect_size=10, char_rep=False, char_vocab_size=None):
        super(HUARN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.stage = stage
        self.char_rep = char_rep
        self.aspect_size = aspect_size
        word_att_size = hidden_size[0]*2
        clause_att_size = hidden_size[1]*2
        self.word_attention = Attention(word_att_size, word_att_size//2).to(device)
        self.clause_attention = Attention(clause_att_size, clause_att_size//2).to(device)
        if word_embedding is not None:
            self.word_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(word_embedding), freeze=False)
            padding_idx = word_embedding.shape[1] - 1
            self.word_embedding.weight.data[padding_idx].uniform_(-0.25, 0.25)
        else:
            self.word_embedding = nn.Embedding(char_vocab_size, input_size, padding_idx=char_vocab_size-1)
        self.word_gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size[0],
            batch_first=True, bidirectional=True)
        self.clause_gru = nn.GRU(
            input_size=hidden_size[0]*2, hidden_size=hidden_size[1],
            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.apply(init_xavier_uniform)
    
    def _init_hidden(self, size, hidden_size):
        hidden = torch.zeros(2, size, hidden_size, device=self.device)
        return hidden

    def forward(self, x, clause_len, text_len, pos, aspect=None):
        word_hidden = self._init_hidden(len(clause_len), self.hidden_size[0])
        clause_len = torch.LongTensor(clause_len).to(self.device).requires_grad_(False)
        x_pack, x_unsort_idx = self.sort_pack(x, clause_len, pos)
        word_out, _ = self.word_gru(x_pack, word_hidden)
        word_out = nn.utils.rnn.pad_packed_sequence(word_out, batch_first=True)
        word_out = word_out[0][x_unsort_idx]
        word_att_inp = word_out
        weighted_word, _ = self.word_attention(word_out, word_att_inp, clause_len)
        weighted_word = self.dropout(weighted_word)
        
        clause_inp = self._extract_clause_inp(weighted_word, text_len)
        clause_hidden = self._init_hidden(clause_inp.size(0), self.hidden_size[1])
        clause_pack, clause_unsort_idx = self.sort_pack(clause_inp, text_len, word=False)
        clause_out, _ = self.clause_gru(clause_pack, clause_hidden)
        clause_out = nn.utils.rnn.pad_packed_sequence(clause_out, batch_first=True)
        clause_out = clause_out[0][clause_unsort_idx]
        clause_att_inp = clause_out
        weighted_clause, _ = self.clause_attention(clause_out, clause_att_inp, text_len.float())
        weighted_clause = self.dropout(weighted_clause)
        return weighted_clause

    def sort_pack(self, x, x_len, pos=None, word=True):
        x_len = np.array(x_len)
        x_sort_idx = np.argsort(-x_len)
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx).to(self.device)]
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx)).to(self.device)
        if word:
            x_emb = self.word_embedding(x)
            if self.char_rep:
                x_emb = x_emb.sum(dim=2)
        else:
            x_emb = x
        x_pack = nn.utils.rnn.pack_padded_sequence(x_emb, x_len, batch_first=True)
        return x_pack, x_unsort_idx

    def _extract_clause_inp(self, weighted_word, text_len):
        clause_inp = []
        start_idx = 0
        for tl in text_len:
            ci = weighted_word[start_idx:start_idx+tl]
            clause_inp.append(ci)
            start_idx += tl
        clause_inp = pad_sequence(clause_inp, batch_first=True)
        return clause_inp
