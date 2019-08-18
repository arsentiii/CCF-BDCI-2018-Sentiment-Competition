import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .init_util import init_xavier_uniform
from model.huarn import HUARN

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

class AHUARN(nn.Module):
    def __init__(self, input_size, hidden_size, aspect_embedding, word_embedding, pos_vocab_size, device, stage, aspect_size=10, char_rep=False, char_vocab_size=None, pinyin=False):
        super(AHUARN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.stage = stage
        self.char_rep = char_rep
        word_att_size = hidden_size[0]*2
        clause_att_size = hidden_size[1]*2
        self.aspect_embedding = aspect_embedding
        if aspect_embedding.shape[1] != input_size:
            aspect_embedding_size = input_size
            self.aspect_embedding = nn.Embedding(aspect_embedding.shape[0], input_size)
        else:
            # aspect_embedding_size = aspect_embedding.shape[1]
            self.aspect_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(aspect_embedding), freeze=False)
            # aspect_embedding_size = aspect_embedding.shape[1]
            # self.aspect_embedding = nn.Embedding.from_pretrained(
            #     torch.FloatTensor(aspect_embedding))
        self.aspect_size = aspect_size
        aspect_embedding_size = input_size
        word_att_size += aspect_embedding_size
        clause_att_size += aspect_embedding_size
        self.word_attention = Attention(word_att_size, word_att_size//2).to(device)
        self.self_word_attention = Attention(hidden_size[0]*2, hidden_size[0])
        self.clause_attention = Attention(clause_att_size, clause_att_size//2).to(device)
        if char_rep is False:
            self.word_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(word_embedding), freeze=False)
            # self.word_embedding = nn.Embedding.from_pretrained(
            #     torch.FloatTensor(word_embedding))
            padding_idx = word_embedding.shape[0] - 1
            self.word_embedding.padding_idx = padding_idx
            self.word_embedding.weight.data[padding_idx].uniform_(-0.25, 0.25)
            self.self_word_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(word_embedding), freeze=False)
            # self.self_word_embedding = nn.Embedding.from_pretrained(
            #     torch.FloatTensor(word_embedding))
            self.self_word_embedding.padding_idx = padding_idx
            self.self_word_embedding.weight.data[padding_idx].uniform_(-0.25, 0.25)           
        else:
            self.word_embedding = nn.Embedding(char_vocab_size, input_size, padding_idx=char_vocab_size-1)
            self.self_word_embedding = nn.Embedding(char_vocab_size, input_size, padding_idx=char_vocab_size-1)
            # for i in range(char_vocab_size):
            #     self.word_embedding.weight.data[i].uniform_(-0.25, 0.25)
            #     self.self_word_embedding.weight.data[i].uniform_(-0.25, 0.25)
        # self.pos_embedding = nn.Embedding(pos_vocab_size, input_size, padding_idx=pos_vocab_size-1)
        self.word_gru = nn.LSTM(
            input_size=input_size+aspect_embedding_size, hidden_size=hidden_size[0],
            batch_first=True, bidirectional=True)
        self.clause_gru = nn.GRU(
            input_size=hidden_size[0]*4, hidden_size=hidden_size[1],
            batch_first=True, bidirectional=True)
        self.self_word_gru = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size[0],
            batch_first=True, bidirectional=True)

        ###0.65034527原先是有下面两个gate
        
        # self.gate1 = nn.Linear(hidden_size[0]*2, 1)
        # self.gate2 = nn.Linear(hidden_size[0]*2, 1)
        self.word_dropout = nn.Dropout(0.5)
        self.clause_dropout = nn.Dropout(0.5)
        self.apply(init_xavier_uniform)

    def _init_hidden(self, size, hidden_size):
        hidden = torch.zeros(2, size, hidden_size).to(self.device)
        # hidden = (torch.zeros(2, size, hidden_size).to(self.device), torch.zeros(2, size, hidden_size).to(self.device))
        return hidden

    def _init_lstm_hidden(self, size, hidden_size):
        hidden = (torch.zeros(2, size, hidden_size).to(self.device), torch.zeros(2, size, hidden_size).to(self.device))
        return hidden

    def forward(self, x, clause_len, text_len, pos, aspect=None):
        aspect_index = torch.LongTensor([i for i in range(self.aspect_size)]).to(self.device)
        aspect_embed = self.aspect_embedding(aspect_index)
        word_target_aspect_embed = torch.matmul(aspect[1], aspect_embed)
        clause_target_aspect_embed = torch.matmul(aspect[0], aspect_embed)
        
        word_hidden = self._init_lstm_hidden(len(clause_len), self.hidden_size[0])
        clause_len = torch.LongTensor(clause_len).to(self.device).requires_grad_(False)
        x_pack, x_unsort_idx = self.sort_pack(x, clause_len, word_target_aspect_embed, text_len, pos)
        word_out, _ = self.word_gru(x_pack, word_hidden)
        word_out = nn.utils.rnn.pad_packed_sequence(word_out, batch_first=True)
        word_out = word_out[0][x_unsort_idx]
        word_aspect = word_target_aspect_embed.unsqueeze(1).expand(-1, word_out.size(1), -1)
        word_att_inp = torch.cat((word_aspect, word_out), -1)
        weighted_word, _ = self.word_attention(word_out, word_att_inp, clause_len)

        self_word_hidden = self._init_lstm_hidden(len(clause_len), self.hidden_size[0])
        self_x_pack, self_x_unsort_idx = self.sort_pack(x, clause_len, None, text_len, pos, if_self=True)
        self_word_out, _ = self.self_word_gru(self_x_pack, self_word_hidden)
        self_word_out = nn.utils.rnn.pad_packed_sequence(self_word_out, batch_first=True)
        self_word_out = self_word_out[0][self_x_unsort_idx]
        self_weighted_word, _ = self.self_word_attention(self_word_out, self_word_out, clause_len)

        gated_weighted_word = torch.cat((weighted_word, self_weighted_word), -1)
        clause_inp = self._extract_clause_inp(gated_weighted_word, text_len)
        clause_hidden = self._init_hidden(clause_inp.size(0), self.hidden_size[1])
        clause_pack, clause_unsort_idx = self.sort_pack(clause_inp, text_len, clause_target_aspect_embed, word=False)
        clause_out, _ = self.clause_gru(clause_pack, clause_hidden)
        clause_out = nn.utils.rnn.pad_packed_sequence(clause_out, batch_first=True)
        clause_out = clause_out[0][clause_unsort_idx]
        cae = clause_target_aspect_embed.unsqueeze(1).expand(-1, clause_out.size(1), -1)
        clause_att_inp = torch.cat((cae, clause_out), -1)
        weighted_clause, _ = self.clause_attention(clause_out, clause_att_inp, text_len.float())
        weighted_clause = self.clause_dropout(weighted_clause)
        return weighted_clause

    def sort_pack(self, x, x_len, apsect_embed, text_len=None, pos=None, word=True, if_self=False):
        x_len = np.array(x_len)
        x_sort_idx = np.argsort(-x_len)
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx).to(self.device)]
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx)).to(self.device)
        if word:
            if if_self:
                x_emb = self.self_word_embedding(x)
            else:
                x_emb = self.word_embedding(x)
            if self.char_rep:
                x_emb = x_emb.sum(dim=2)
        else:
            x_emb = x
        if not if_self and word:
            a_emb = apsect_embed.unsqueeze(1).expand(-1, x_emb.size(1), -1)
            emb = torch.cat((x_emb, a_emb), -1)
        else:
            emb = x_emb
        if not word:
            emb = self.word_dropout(emb)
        x_pack = nn.utils.rnn.pack_padded_sequence(emb, x_len, batch_first=True)
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
