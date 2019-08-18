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

class HUARN(nn.Module):
    def __init__(self, input_size, hidden_size, aspect_embedding, word_embedding, device, stage, aspect_size=10):
        super(HUARN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.stage = stage
        word_att_size = hidden_size[0]*2
        clause_att_size = hidden_size[1]*2
        if stage == 2:
            self.aspect_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(aspect_embedding))
            self.aspect_size = aspect_size
            word_att_size += aspect_embedding.shape[1]
            clause_att_size += aspect_embedding.shape[1]
        self.word_attention = Attention(word_att_size, word_att_size//2).to(device)
        self.clause_attention = Attention(clause_att_size, clause_att_size//2).to(device)
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding))
        self.word_gru = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size[0],
            batch_first=True, bidirectional=True)
        self.clause_gru = nn.LSTM(
            input_size=hidden_size[0]*2, hidden_size=hidden_size[1],
            batch_first=True, bidirectional=True)
        self.apply(init_xavier_uniform)
    
    def _init_hidden(self, size, hidden_size):
        hidden = (torch.zeros(2, size, hidden_size).to(self.device), torch.zeros(2, size, hidden_size).to(self.device))
        return hidden

    def forward(self, x, clause_len, text_len, aspect=None):
        if self.stage == 2:
            aspect_index = torch.LongTensor([i for i in range(self.aspect_size)]).to(self.device)
            aspect_embed = self.aspect_embedding(aspect_index)
            target_aspect_embed = torch.matmul(aspect, aspect_embed)
        word_hidden = self._init_hidden(len(clause_len), self.hidden_size[0])
        clause_len = torch.LongTensor(clause_len).to(self.device).requires_grad_(False)
        x_pack, x_unsort_idx = self.sort_pack(x, clause_len)
        word_out, _ = self.word_gru(x_pack, word_hidden)
        word_out = nn.utils.rnn.pad_packed_sequence(word_out, batch_first=True)
        word_out = word_out[0][x_unsort_idx]
        if self.stage == 1:
            word_att_inp = word_out
        else:
            word_aspect_inp = []
            for i, size in enumerate(text_len):
                wae = target_aspect_embed[i].expand(size, -1)
                wae = wae.unsqueeze(1).expand(-1, word_out.size(1), -1)
                word_aspect_inp.append(wae)
            word_aspect = torch.cat(word_aspect_inp, 0)
            word_att_inp = torch.cat((word_aspect, word_out), -1)

        weighted_word, _ = self.word_attention(word_out, word_att_inp, clause_len)
        weighted_word = F.dropout(weighted_word, 0.5)
        clause_inp = self._extract_clause_inp(weighted_word, text_len)
        clause_hidden = self._init_hidden(clause_inp.size(0), self.hidden_size[1])
        clause_pack, clause_unsort_idx = self.sort_pack(clause_inp, text_len, word=False)
        clause_out, _ = self.clause_gru(clause_pack, clause_hidden)
        clause_out = nn.utils.rnn.pad_packed_sequence(clause_out, batch_first=True)
        clause_out = clause_out[0][clause_unsort_idx]
        if self.stage == 1:
            clause_att_inp = clause_out
        else:
            cae = target_aspect_embed.unsqueeze(1).expand(-1, clause_out.size(1), -1)
            clause_att_inp = torch.cat((cae, clause_out), -1)
        weighted_clause, _ = self.clause_attention(clause_out, clause_att_inp, text_len.float())
        # if self.stage == 2:
            # weighted_clause = weighted_clause + target_aspect_embed
        if self.stage == 2:
            weighted_clause = torch.cat((weighted_clause, target_aspect_embed), -1)
        weighted_clause = F.dropout(weighted_clause, 0.5)
        return weighted_clause

    def sort_pack(self, x, x_len, word=True):
        x_len = np.array(x_len)
        x_sort_idx = np.argsort(-x_len)
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx).to(self.device)]
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx)).to(self.device)
        if word:
            x_emb = self.word_embedding(x)
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