import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .init_util import *

class CharNN(nn.Module):
    def __init__(self, input_size, hidden_size, conv_num, aspect_embedding, char_vocab_size, device, stage, aspect_size=10, word=False, input_embedding=None, pos_vocab_size=None):
        super(CharNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_num = conv_num
        self.device = device
        self.stage = stage
        self.word = word
        char_att_size = input_size
        if stage == 2:
            if aspect_embedding is not None:
                self.aspect_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(aspect_embedding))
                char_att_size += aspect_embedding.shape[1]
            else:
                self.aspect_embedding = nn.Embedding(aspect_size, input_size)
                char_att_size += input_size
        self.char_attention = Attention(char_att_size, char_att_size//2).to(device)
        char_padding_idx = char_vocab_size - 1
        if word:
            self.char_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(input_embedding))
            char_padding_idx = input_embedding.shape[0] - 1
            self.char_embedding.padding_idx = char_padding_idx
            self.char_embedding.weight.data[char_padding_idx].uniform_(-0.25, 0.25)
        else:
            self.char_embedding = nn.Embedding(char_vocab_size, input_size, padding_idx=char_padding_idx)
        self.char_gru = nn.GRU(
            input_size=conv_num, hidden_size=hidden_size,
            batch_first=True, bidirectional=True)
        self.conv = nn.Conv2d(1, conv_num, (3, input_size), padding=(1, 0))
        self.dropout = nn.Dropout(0.5)
        self.apply(init_xavier_uniform)
    
    def _init_hidden(self, size, hidden_size):
        # hidden = (torch.zeros(2, size, hidden_size, device=self.device), torch.zeros(2, size, hidden_size, device=self.device))
        hidden = torch.zeros(2, size, hidden_size, device=self.device)
        return hidden

    def forward(self, chars, char_len, pos=None):
        char_emb = self.char_embedding(chars).unsqueeze(1)
        conv_char = self.conv(char_emb).squeeze(3).permute(0, 2, 1)
        conv_char = torch.relu(conv_char)
        hidden = self._init_hidden(conv_char.size(0), self.hidden_size)
        x_pack, x_unsort_idx = self.sort_pack(conv_char, char_len)
        char_out, _ = self.char_gru(x_pack, hidden)
        char_out = nn.utils.rnn.pad_packed_sequence(char_out, batch_first=True)
        char_out = char_out[0][x_unsort_idx]
        weighted_char, _ = self.char_attention(char_out, char_out, char_len)
        weighted_char = self.dropout(weighted_char)
        return weighted_char

    def sort_pack(self, x, x_len):
        x_len = np.array(x_len)
        x_sort_idx = np.argsort(-x_len)
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx).to(self.device)]
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx)).to(self.device)
        x_pack = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
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
