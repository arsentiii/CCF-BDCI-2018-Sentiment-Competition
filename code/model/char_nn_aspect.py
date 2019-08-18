import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .init_util import init_xavier_uniform

class Attention(nn.Module):
    def __init__(self, hid_size, att_size):
        super(Attention, self).__init__()
        self.lin = nn.Linear(hid_size, att_size)
        self.att_w = nn.Linear(att_size, 1, bias=False)
        self.apply(init_xavier_uniform)

    def forward(self, enc_sents, enc_sent_com_attent, len_s):
        emb_h = torch.tanh(self.lin(enc_sent_com_attent))
        att = self.att_w(emb_h).squeeze(-1)
        att_soft_res = self._masked_softmax(att, len_s)
        out = att_soft_res.unsqueeze(-1)
        attended = out * enc_sents
        return attended.sum(1, True).squeeze(1), att_soft_res

    def _masked_softmax(self, mat, len_s):
        len_s = len_s.long()
        idxes = torch.arange(0, int(len_s.max()), out=mat.data.new(int(len_s.max())).long()).unsqueeze(1)
        mask = (idxes < len_s.unsqueeze(0)).float().permute(1, 0).requires_grad_(False)
        exp = torch.exp(mat) * mask
        sum_exp = exp.sum(1, True) + 0.0001
        return exp / sum_exp.expand_as(exp)

class CharNN(nn.Module):
    def __init__(self, input_size, hidden_size, conv_num, aspect_embedding, char_vocab_size, device, stage, aspect_size=10, word=False, word_embedding=None):
        super(CharNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_num = conv_num
        self.device = device
        self.stage = stage
        self.aspect_size = aspect_size
        char_att_size = conv_num + input_size
        # self.aspect_embedding = aspect_embedding
        # if aspect_embedding is not None:
        self.aspect_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(aspect_embedding), freeze=False)
        # self.aspect_embedding = nn.Embedding.from_pretrained(
        #     torch.FloatTensor(aspect_embedding))
        #     char_att_size += aspect_embedding.shape[1]
        # else:
        # self.aspect_embedding = nn.Embedding(aspect_size, input_size)
        #     char_att_size += input_size
        self.char_attention = Attention(char_att_size, char_att_size//2)
        # self.self_char_attention = Attention(conv_num, conv_num//2)
        if word is False:
            char_padding_idx = char_vocab_size - 1
            self.char_embedding = nn.Embedding(char_vocab_size, input_size)
            # for i in range(char_vocab_size):
            #     self.char_embedding.weight.data[i].uniform_(-0.25, 0.25)
            # self.self_char_embedding = nn.Embedding(char_vocab_size, input_size, padding_idx=char_padding_idx)
        else:
            char_padding_idx = word_embedding.shape[0] - 1
            self.char_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(word_embedding), freeze=False)
            # self.char_embedding = nn.Embedding.from_pretrained(
            #     torch.FloatTensor(word_embedding))
            self.char_embedding.weight.data[char_padding_idx].uniform_(-0.25, 0.25)
            ## 0.65222850 有下面的embedding 
            # self.self_char_embedding = nn.Embedding.from_pretrained(
            #     torch.FloatTensor(word_embedding))
            # self.self_char_embedding.weight.data[char_padding_idx].uniform_(-0.25, 0.25)
        # self.char_embedding.weight.data[char_padding_idx].uniform_(-0.25, 0.25)
        self.char_gru = nn.LSTM(
            input_size=conv_num, hidden_size=hidden_size,
            batch_first=True, bidirectional=True)
        self.conv = nn.Conv2d(1, conv_num, (3, input_size * 2), padding=(1, 0))
        # self.self_char_gru = nn.GRU(
        #     input_size=conv_num, hidden_size=hidden_size,
        #     batch_first=True, bidirectional=True)
        # self.self_conv = nn.Conv2d(1, conv_num, (3, input_size), padding=(1, 0))
        # self.gate1 = nn.Linear(conv_num, 1)
        # self.gate2 = nn.Linear(conv_num, 1)    
        self.dropout = nn.Dropout(0.5)
        self.apply(init_xavier_uniform)
    
    def _init_hidden(self, size, hidden_size):
        # hidden = torch.zeros(2, size, hidden_size, device=self.device)
        hidden = (torch.zeros(2, size, hidden_size, device=self.device), torch.zeros(
            2, size, hidden_size, device=self.device))
        return hidden

    def forward(self, chars, char_len, aspect=None):
        char_emb = self.char_embedding(chars).unsqueeze(1)
        aspect_index = torch.LongTensor([i for i in range(self.aspect_size)]).to(self.device)
        aspect_embed = self.aspect_embedding(aspect_index)
        target_aspect_embed = torch.matmul(aspect[0], aspect_embed).unsqueeze(1)
        aspcet_emb = target_aspect_embed.unsqueeze(1).expand_as(char_emb)
        aspect_char_emb = torch.cat((char_emb, aspcet_emb), -1)
        conv_char = self.conv(aspect_char_emb).squeeze(3).permute(0, 2, 1)
        conv_char = torch.relu(conv_char)
        hidden = self._init_hidden(conv_char.size(0), self.hidden_size)
        x_pack, x_unsort_idx = self.sort_pack(conv_char, char_len)
        char_out, _ = self.char_gru(x_pack, hidden)
        char_out = nn.utils.rnn.pad_packed_sequence(char_out, batch_first=True)
        char_out = char_out[0][x_unsort_idx]
        char_att_inp = target_aspect_embed.expand(-1, char_out.size(1), -1)
        char_out_inp = torch.cat((char_out, char_att_inp), -1)
        weighted_char, _ = self.char_attention(char_out, char_out_inp, char_len)
        
        # self_char_emb = self.self_char_embedding(chars).unsqueeze(1)
        # self_conv_char = self.self_conv(self_char_emb).squeeze(3).permute(0, 2, 1)
        # self_conv_char = torch.relu(self_conv_char)
        # self_hidden = self._init_hidden(self_conv_char.size(0), self.hidden_size)
        # self_x_pack, self_x_unsort_idx = self.sort_pack(self_conv_char, char_len)
        # self_char_out, _ = self.char_gru(self_x_pack, self_hidden)
        # self_char_out = nn.utils.rnn.pad_packed_sequence(self_char_out, batch_first=True)
        # self_char_out = self_char_out[0][self_x_unsort_idx]
        # self_weighted_char, _ = self.self_char_attention(self_char_out, self_char_out, char_len)
        # self_weighted_char = self.dropout(self_weighted_char)
        # h = torch.sigmoid(self.gate1(weighted_char) + self.gate2(self_weighted_char))
        # r = torch.ones_like(h)
        # h = torch.max(r*0.8, h)
        # h = torch.min(r*0.2, h)
        # gated_weighted_char = h * weighted_char + (1. - h) * self_weighted_char
        gated_weighted_char = self.dropout(weighted_char)
        return gated_weighted_char

    def sort_pack(self, x, x_len):
        x_len = np.array(x_len)
        x_sort_idx = np.argsort(-x_len)
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx, device=self.device)]
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx), device=self.device)
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
