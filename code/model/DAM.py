import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .init_util import init_xavier_uniform

class FFN(nn.Module):
	def __init__(self, input_size):
		super(FFN, self).__init__()
		self.w1 = nn.Linear(input_size, input_size)
		self.w2 = nn.Linear(input_size, input_size)
		self.apply(init_xavier_uniform)

	def forward(self, x):
		o1 = torch.relu(self.w1(x))
		o2 = torch.relu(self.w2(o1))
		o = o2 + x
		return o

class AttentiveModule(nn.Module):
	def __init__(self, input_size, device):
		super(AttentiveModule, self).__init__()
		self.input_size = input_size
		self.device = device
		self.ffn = FFN(input_size)

	def forward(self, inp, inp_len, aspect):
		scale_factor = inp_len.sqrt().unsqueeze(1).unsqueeze(2)
		att = torch.matmul(aspect, inp.permute(0, 2, 1)) / scale_factor
		att_soft_res, mask = self._masked_softmax(att, inp_len)
		weighted_inp = torch.matmul(att_soft_res, inp)
		mask = mask.unsqueeze(-1).expand_as(weighted_inp)
		ffn_inp = weighted_inp + (inp + aspect) * mask
		ffn_out = self.ffn(ffn_inp)
		final_out = ffn_inp + ffn_out
		return final_out / torch.norm(final_out, 2, 2, True)

	def _masked_softmax(self, mat, len_s):
		len_s = len_s.long()
		idxes = torch.arange(0, int(len_s.max()), out=mat.data.new(int(len_s.max())).long()).unsqueeze(1)
		mask = (idxes < len_s.unsqueeze(0)).float().permute(1, 0).requires_grad_(False)
		expand_mask = mask.unsqueeze(-1).expand_as(mat)
		exp = mat * expand_mask
		sum_exp = exp.sum(-1, True) + 0.0001
		return exp / sum_exp.expand_as(exp), mask

class DAM(nn.Module):
	def __init__(self, input_size, word_embedding, aspect_embedding, aspect_size, device):
		super(DAM, self).__init__()
		self.input_size = input_size
		self.aspect_size = aspect_size
		self.device = device
		self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding))
		padding_idx = word_embedding.shape[0] - 1
		self.word_embedding.padding_idx = padding_idx
		self.word_embedding.weight.data[padding_idx].uniform_(-0.25, 0.25)
		if aspect_embedding is not None:
			self.aspect_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(aspect_embedding))
		self.word_att = nn.ModuleList([AttentiveModule(input_size, device) for _ in range(2)])
		self.clause_att = nn.ModuleList([AttentiveModule(input_size, device) for _ in range(2)])

	def forward(self, x, clause_len, text_len, pos, aspect=None):
		word_inp = self.word_embedding(x)
		aspect_index = torch.LongTensor([i for i in range(self.aspect_size)]).to(self.device)
		aspect_embed = self.aspect_embedding(aspect_index)
		word_target_aspect_embed = torch.matmul(aspect[1], aspect_embed).unsqueeze(1).expand_as(word_inp)
		inp_len = torch.Tensor(clause_len).to(self.device)
		for i in range(2):
			word_att_out = self.word_att[i](word_inp, inp_len, word_target_aspect_embed)
			word_inp = word_att_out

		clause_inp = self._extract_clause_inp(word_att_out.sum(dim=1), text_len)
		clause_target_aspect_embed = torch.matmul(aspect[0], aspect_embed).unsqueeze(1).expand_as(clause_inp)
		for i in range(2):
			clause_att_out = self.clause_att[i](clause_inp, text_len.float(), clause_target_aspect_embed)
			clause_inp = clause_att_out
		final_out = clause_att_out.sum(dim=1)
		return final_out

	def _extract_clause_inp(self, weighted_word, text_len):
		clause_inp = []
		start_idx = 0
		for tl in text_len:
			ci = weighted_word[start_idx:start_idx+tl]
			clause_inp.append(ci)
			start_idx += tl
		clause_inp = pad_sequence(clause_inp, batch_first=True)
		return clause_inp