import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.huarn2 import HUARN
from model.ahuarn import AHUARN
from model.char_nn_aspect import CharNN
from model.highway import HighWay
from model.DAM import DAM
from model.word_nn import WordNN
import gc
from .init_util import init_xavier_uniform
import torch.nn.utils.weight_norm as wm


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
        idxes = torch.arange(0, int(len_s.max()), out=mat.data.new(
            int(len_s.max())).long()).unsqueeze(1)
        mask = (idxes < len_s.unsqueeze(0)).float().permute(
            1, 0).requires_grad_(False)
        exp = torch.exp(mat) * mask
        sum_exp = exp.sum(1, True) + 0.0001
        return exp / sum_exp.expand_as(exp)


class SentimentNet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 aspect_size, tag_size, aspect_embedding, word_embedding,
                 word_pinyin_embedding, pos_vocab_size,
                 char_nn_params, pinyin_nn_params, device):
        super(SentimentNet, self).__init__()
        self.aspect_size = aspect_size
        self.tag_size = tag_size
        self.device = device
        self.aspect_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(aspect_embedding), freeze=False)
        self.HUARN = AHUARN(input_size, hidden_size, aspect_embedding,
                            word_embedding, pos_vocab_size, device, stage=2)
        self.HUARN_pinyin_rep = AHUARN(input_size, hidden_size,
                                       aspect_embedding, word_pinyin_embedding, pos_vocab_size, device, stage=2, pinyin=True)
        char_nn_params[3] = aspect_embedding
        self.char_encoder = CharNN(*char_nn_params)
        self.word_encoder = WordNN(
            input_size, hidden_size[0], aspect_embedding, word_embedding, device)
        self.conv = nn.ModuleList([
            nn.Conv2d(1, 300, (i, 300)) for i in (2, 3, 4)
        ])
        self.highway1 = HighWay(1200, 900)
        self.sentiment_decoder = nn.Linear(2100, tag_size)
        self.dropout = nn.Dropout(0.5)
        self.apply(init_xavier_uniform)

    def forward(self, x, aspect, clause_len,
                text_len, idxs, position, pos,
                chars, char_len, clause_chars, word_pinyin,
                word_pinyin_one_level, word_pinyin_len, x_one_level, text_len_one_level):
        aspect_index = torch.LongTensor(
            [i for i in range(self.aspect_size)]).to(self.device)
        aspect_embed = self.aspect_embedding(aspect_index)
        aspect_emb = torch.matmul(aspect[0], aspect_embed)
        encoder_word = self.HUARN(x, clause_len, text_len, pos, aspect)
        encoder_word_char_rep = self.HUARN_char_rep(
            clause_chars, clause_len, text_len, pos, aspect)
        encoder_char = self.char_encoder(chars, char_len, aspect)
        encoder_word_pinyin = self.HUARN_pinyin_rep(
            word_pinyin, clause_len, text_len, pos, aspect)
        encoder_word_one = self.word_encoder(
            x_one_level, text_len_one_level, aspect)
        conv_inp = torch.cat((
            encoder_word_pinyin.unsqueeze(1).unsqueeze(1),
            encoder_char.unsqueeze(1).unsqueeze(1),
            encoder_word.unsqueeze(1).unsqueeze(1),
            # encoder_word_char_rep.unsqueeze(1).unsqueeze(1),
            encoder_word_one.unsqueeze(1).unsqueeze(1)
        ), 2)
        convz = [torch.relu(conv(conv_inp).squeeze(3)) for conv in self.conv]
        max_convz = [F.max_pool1d(cz, cz.size(2)).squeeze(2) for cz in convz]
        convzz = torch.cat(max_convz, -1)
        z1 = self.highway1((encoder_word,
                            encoder_char, encoder_word_pinyin, encoder_word_one))
        z = torch.cat((z1, convzz, aspect_emb), -1)
        # z = torch.cat((z1, z2, z3, z4, z5, z6, aspect_emb), -1)
        encoder_x = self.dropout(z)
        logits = self.sentiment_decoder(encoder_x)
        return logits
