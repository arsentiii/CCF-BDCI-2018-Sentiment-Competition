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

class WordNN(nn.Module):
    def __init__(self, input_size, hidden_size, aspect_embedding, word_embedding, device, aspect_size=10):
        super(WordNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.aspect_size = aspect_size
        word_att_size = aspect_embedding.shape[1] + input_size
        self.aspect_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(aspect_embedding), freeze=False)
        # self.aspect_embedding = nn.Embedding.from_pretrained(
        #     torch.FloatTensor(aspect_embedding))
        self.word_attention = Attention(word_att_size, word_att_size//2)
        self.self_word_attention = Attention(input_size, input_size//2)
        padding_idx = word_embedding.shape[0] - 1
        self.word_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(word_embedding), freeze=False)
        # self.word_embedding = nn.Embedding.from_pretrained(
        #     torch.FloatTensor(word_embedding))
        self.word_embedding.weight.data[padding_idx].uniform_(-0.25, 0.25)
        # self.self_word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding), freeze=False)
        self.self_word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding))
        self.self_word_embedding.weight.data[padding_idx].uniform_(-0.25, 0.25)
        self.word_gru = nn.LSTM(
            input_size=input_size+aspect_embedding.shape[1], hidden_size=hidden_size,
            batch_first=True, bidirectional=True)
        self.self_word_gru = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            batch_first=True, bidirectional=True)
        self.project = nn.Linear(hidden_size*4, hidden_size*2)
        self.dropout = nn.Dropout(0.5)
        self.apply(init_xavier_uniform)
    
    def _init_hidden(self, size, hidden_size):
        # hidden = torch.zeros(2, size, hidden_size).to(self.device)
        hidden = (torch.zeros(2, size, hidden_size).to(self.device), torch.zeros(2, size, hidden_size).to(self.device))
        return hidden

    def forward(self, x, x_len, aspect=None):
        aspect_index = torch.LongTensor([i for i in range(self.aspect_size)]).to(self.device)
        aspect_embed = self.aspect_embedding(aspect_index)
        target_aspect_embed = torch.matmul(aspect[0], aspect_embed)

        hidden = self._init_hidden(x.size(0), self.hidden_size)
        x_pack, x_unsort_idx = self.sort_pack(x, x_len, target_aspect_embed, False, 0)
        word_out, _ = self.word_gru(x_pack, hidden)
        word_out = nn.utils.rnn.pad_packed_sequence(word_out, batch_first=True)
        word_out = word_out[0][x_unsort_idx]
        aspect_inp = target_aspect_embed.unsqueeze(1).expand_as(word_out)
        word_att_inp = torch.cat((word_out, aspect_inp), -1)
        weighted_word, _ = self.word_attention(word_out, word_att_inp, x_len)

        self_hidden = self._init_hidden(x.size(0), self.hidden_size)
        self_x_pack, self_x_unsort_idx = self.sort_pack(x, x_len, target_aspect_embed, False, 0)
        self_word_out, _ = self.word_gru(self_x_pack, self_hidden)
        self_word_out = nn.utils.rnn.pad_packed_sequence(self_word_out, batch_first=True)
        self_word_out = self_word_out[0][self_x_unsort_idx]
        self_weighted_word, _ = self.self_word_attention(self_word_out, self_word_out, x_len)        

        final_word = torch.cat((weighted_word, self_weighted_word), -1)
        out = self.project(final_word)
        out = torch.relu(out)
        out = self.dropout(out)
        return out

    def sort_pack(self, x, x_len, apsect_embed, if_self, level):
        x_len = np.array(x_len)
        x_sort_idx = np.argsort(-x_len)
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx).to(self.device)]
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx)).to(self.device)
        if level == 0:
            if if_self:
                x_emb = self.self_word_embedding(x)
            else:
                x_emb = self.word_embedding(x)
                a_emb = apsect_embed.unsqueeze(1).expand(-1, x_emb.size(1), -1)
                x_emb = torch.cat((x_emb, a_emb), -1)
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
