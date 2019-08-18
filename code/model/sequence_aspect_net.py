import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.huarn import HUARN
from .init_util import init_xavier_uniform
from model.char_nn import CharNN
from model.highway import HighWay
from model.word_nn_for_aspect import WordNN

class SeqDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, aspect_size, device):
        super(SeqDecoder, self).__init__()
        self.label_embedding = nn.Embedding(11, hidden_size, padding_idx=0)
        self.hidden_size = hidden_size
        self.aspect_size = aspect_size
        self.device = device
        self.gru = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.linear = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(self.aspect_size)
        ])
        self.w1 = nn.Linear(self.hidden_size, 1, bias=False)
        self.w2 = nn.Linear(self.hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.apply(init_xavier_uniform)

    def _init_hidden(self, size):
        hidden = torch.zeros(size, self.hidden_size).to(self.device)
        return hidden

    def forward(self, encoder_x):
        hidden = self._init_hidden(encoder_x.size(0))
        g = 0.0
        logit = []
        prob = []
        sum_e = [torch.zeros(encoder_x.size(0), 1, self.hidden_size).to(self.device)]
        for i in range(self.aspect_size):
            inp = encoder_x + g
            hidden = self.gru(inp, hidden)
            hidden = self.dropout(hidden)
            logi = self.linear[i](hidden)
            p = torch.sigmoid(logi)
            se = torch.sum(torch.cat(sum_e, 1), 1)
            e = self.label_embedding(torch.LongTensor([i+1]).to(self.device)).expand(encoder_x.size(0), -1)
            h = torch.sigmoid(self.w1(se) + self.w2(e))
            pred = ((p > 0.5).long() * (i+1))
            cur_e = self.label_embedding(pred)
            sum_e.append(cur_e)
            g = (1 - h) * e + h * se
            logit.append(logi)
            prob.append(p)
        logits = torch.cat(logit, -1)
        probs = torch.cat(prob, -1)
        return logits, probs

class SequenceAspectNet(nn.Module):
    def __init__(self, input_size, hidden_size, aspect_size, word_embedding, word_pinyin_embedding, pos_vocab_size, char_nn_params, device):
        super(SequenceAspectNet, self).__init__()
        self.HUARN = HUARN(input_size, hidden_size, None, word_embedding, pos_vocab_size, device, stage=1)
        self.HUARN_char_rep = HUARN(input_size, hidden_size, None, None, pos_vocab_size, device, stage=1, char_rep=True, char_vocab_size=char_nn_params[-3])
        self.HUARN_pinyin = HUARN(input_size, hidden_size, None, word_pinyin_embedding, pos_vocab_size, device, stage=1)
        self.char_nn = CharNN(*char_nn_params)
        self.word_nn = WordNN(input_size, hidden_size[0], word_embedding, device)
        self.highway1 = HighWay(1500, 900)
        self.conv = nn.ModuleList([
            nn.Conv2d(1, 300, (i, 300)) for i in (2, 3, 4)
        ])
        self.seq_decoder = SeqDecoder(1800, 1800, aspect_size, device)
        self.dropout = nn.Dropout(0.5)
        self.apply(init_xavier_uniform)

    def forward(self, x, clause_len, \
        text_len, pos, chars, char_len,\
        new_clause_chars, word_pinyin, \
        x_one_level, text_len_one_level):
        encoder_word = self.HUARN(x, clause_len, text_len, pos)
        encoder_word_char_rep = self.HUARN_char_rep(new_clause_chars, clause_len, text_len, pos)
        encoder_char = self.char_nn(chars, char_len)
        encoder_word_pinyin = self.HUARN_pinyin(word_pinyin, clause_len, text_len, pos)
        encoder_word_one = self.word_nn(x_one_level, text_len_one_level)
        conv_inp = torch.cat((
            encoder_word_pinyin.unsqueeze(1).unsqueeze(1),
            encoder_char.unsqueeze(1).unsqueeze(1),
            encoder_word.unsqueeze(1).unsqueeze(1),
            encoder_word_char_rep.unsqueeze(1).unsqueeze(1),
            encoder_word_one.unsqueeze(1).unsqueeze(1) 
        ), 2)
        convz = [torch.relu(conv(conv_inp).squeeze(3)) for conv in self.conv]
        max_convz = [F.max_pool1d(cz, cz.size(2)).squeeze(2) for cz in convz]
        convzz = torch.cat(max_convz, -1)
        z1 = self.highway1((encoder_word, encoder_word_char_rep, \
            encoder_char, encoder_word_pinyin, encoder_word_one))
        z = torch.cat((z1, convzz), -1)
        encoder_x = self.dropout(z)
        logits, prob = self.seq_decoder(encoder_x)
        return logits, prob
