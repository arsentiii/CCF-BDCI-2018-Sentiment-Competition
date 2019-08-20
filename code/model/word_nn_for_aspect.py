import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .init_util import *

class WordNN(nn.Module):
    def __init__(self, input_size, hidden_size, word_embedding, device):
        super(WordNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.word_attention = Attention(input_size, input_size//2)
        aspect_size = 10
        self.aspect_size = 10
        padding_idx = word_embedding.shape[0] - 1
        self.word_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(word_embedding), freeze=False)
        self.word_embedding.weight.data[padding_idx].uniform_(-0.25, 0.25)
        self.word_gru = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.apply(init_xavier_uniform)
    
    def _init_hidden(self, size, hidden_size):
        hidden = (torch.zeros(2, size, hidden_size, device=self.device), torch.zeros(2, size, hidden_size, device=self.device))
        return hidden

    def forward(self, x, x_len):
        hidden = self._init_hidden(x.size(0), self.hidden_size)
        x_pack, x_unsort_idx = self.sort_pack(x, x_len)
        word_out, _ = self.word_gru(x_pack, hidden)
        word_out = nn.utils.rnn.pad_packed_sequence(word_out, batch_first=True)
        word_out = word_out[0][x_unsort_idx]
        weighted_word, _ = self.word_attention(word_out, word_out, x_len)
        out = self.dropout(weighted_word)
        return out

    def sort_pack(self, x, x_len):
        x_len = np.array(x_len)
        x_sort_idx = np.argsort(-x_len)
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx).to(self.device)]
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx)).to(self.device)
        x_emb = self.word_embedding(x)
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
