import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.huarn import HUARN
from model.char_nn import CharNN
from model.highway import HighWay
from model.word_nn_for_aspect import WordNN
from .init_util import init_xavier_uniform

class AspectNet(nn.Module):
    def __init__(self, input_size, hidden_size, aspect_size, word_embedding, word_pinyin_embedding, pos_vocab_size, char_nn_params, device):
        super(AspectNet, self).__init__()
        self.HUARN = HUARN(input_size, hidden_size, None, word_embedding, pos_vocab_size, device, stage=1)
        self.HUARN_char_rep = HUARN(input_size, hidden_size, None, None, pos_vocab_size, device, stage=1, char_rep=True, char_vocab_size=char_nn_params[-3])
        self.HUARN_pinyin = HUARN(input_size, hidden_size, None, word_pinyin_embedding, pos_vocab_size, device, stage=1)
        self.char_nn = CharNN(*char_nn_params)
        self.word_nn = WordNN(input_size, hidden_size[0], word_embedding, device)
        self.highway1 = HighWay(1500, 900)
        self.conv = nn.ModuleList([
            nn.Conv2d(1, 300, (i, 300)) for i in (2, 3, 4)
        ])
        self.decoder = nn.Linear(1800, aspect_size)
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
        logits = self.decoder(encoder_x)
        prob = torch.sigmoid(logits)
        return logits, prob
