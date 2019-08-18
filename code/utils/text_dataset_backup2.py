import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class TextData(Dataset):
	def __init__(self, df, test=False, aspect_size=10, 
		padding_value=13690, pos_padding_value=30, 
		char_padding_value=2848, max_word_char_len=4,
		word_pinyin_padding_value=3677, char_pinyin_padding_value=65):
		self.sentiment_hash = {-1:0, 0:1, 1:2}
		self.df = df
		self.aspects = []
		self.data = []
		self.data_one_level = []
		self.aspect_size = aspect_size
		self.clause_len = []
		self.article_len = []
		self.word_len = []
		self.labels = []
		self.char = []
		self.char_len = []
		self.clause_chars = []
		max_clause_len = 0
		max_char_len = 0
		max_word_len = 0
		max_pinyin_len = 0
		self.weight = [0]*10
		self.test = test
		self.idxs = []
		for i, idx, content, aspect, chars, clause_chars, \
		content_one_level \
		in zip(df['content_seg_idx'].index, df['content_id'], \
			df['content_seg_idx'], df['aspect_idx'], \
			df['char_idx'], df['clause_char_idx'], df['content_seg_idx_one_level']):
			clauses = content.split('\t')
			clauses_char = clause_chars.split('\t')
			clause_idx = []
			pos_idx = []
			l = []
			clause_char_idx = []
			clause_word_pinyin_idx = []
			for clause, cc \
			in zip(clauses, clauses_char):
				zzz = [int(idx) for idx in clause.split(' ')]
				cczzz = [[int(iii) if iii != '' else char_padding_value for iii in idx.split('|')][:max_word_char_len] for idx in cc.split(' ')]
				for b in range(len(cczzz)):
					if len(cczzz[b]) < max_word_char_len:
						cczzz[b].extend([char_padding_value for _ in range(max_word_char_len-len(cczzz[b]))])
				zzz_len = len(zzz)
				l.append(str(zzz_len))
				clause_idx.append(zzz)
				clause_char_idx.append(cczzz)
				if zzz_len > max_clause_len:
					max_clause_len = zzz_len
			asp = [int(aidx) for aidx in aspect.split(',')]
			asp = self._generate_binary_label(asp)
			if test is False:
				senti_value = df['sentiment_value'][i]
				senti_v = [self.sentiment_hash[int(z)] for z in senti_value.split(',')]
				label = self._generate_binary_label_sentiment(asp, senti_v)
				self.labels.append(label)
			self.clause_len.append(','.join(l))
			char_seg = [int(idx) for idx in chars.split(' ')]
			char_len = len(char_seg)
			if char_len > max_char_len:
				max_char_len = char_len
			word_one_level = [int(idx) for idx in content_one_level.split(' ')]
			word_len = len(word_one_level)
			if word_len > max_word_len:
				max_word_len = word_len
			self.data_one_level.append(word_one_level)
			self.word_len.append(word_len)
			self.clause_chars.append(clause_char_idx)
			self.char.append(char_seg)
			self.char_len.append(char_len)
			self.article_len.append(len(l))
			self.data.append(clause_idx)
			self.aspects.append(asp)
			self.idxs.append(idx)
			

		for i in range(len(self.data)):
			for j in range(len(self.data[i])):
				clause_len = len(self.data[i][j])
				if clause_len < max_clause_len:
					self.data[i][j].extend([padding_value for _ in range(max_clause_len-clause_len)])
					self.clause_chars[i][j].extend([[char_padding_value]*max_word_char_len for _ in range(max_clause_len-clause_len)])
			if len(self.char[i]) < max_char_len:
				self.char[i].extend([char_padding_value for _ in range(max_char_len-len(self.char[i]))])
			if self.word_len[i] < max_word_len:
				self.data_one_level[i].extend([padding_value for _ in range(max_word_len-self.word_len[i])])
			self.data[i] = torch.LongTensor(self.data[i])
			self.char[i] = torch.LongTensor(self.char[i])
			self.clause_chars[i] = torch.LongTensor(self.clause_chars[i])
			self.data_one_level[i] = torch.LongTensor(self.data_one_level[i])
		self.data = pad_sequence(self.data, batch_first=True, padding_value=padding_value)
		self.char = pad_sequence(self.char, batch_first=True, padding_value=char_padding_value)
		self.clause_chars = pad_sequence(self.clause_chars, batch_first=True, padding_value=char_padding_value)
		self.data_one_level = pad_sequence(self.data_one_level, batch_first=True, padding_value=padding_value)

	def __getitem__(self, index):
		if self.test:
			return self.idxs[index], self.data[index], \
			torch.Tensor(self.aspects[index]), self.clause_len[index], \
			self.article_len[index], self.char[index], \
			self.char_len[index], self.clause_chars[index], \
			self.data_one_level[index], self.word_len[index]
		else:
			return self.idxs[index], self.data[index], \
			torch.Tensor(self.aspects[index]), torch.LongTensor(self.labels[index]), \
			self.clause_len[index], self.article_len[index], \
			self.char[index], \
			self.char_len[index], self.clause_chars[index], \
			self.data_one_level[index], self.word_len[index]

	def __len__(self):
		return len(self.data)

	def _generate_binary_label(self, aspects):
		label = [0] * self.aspect_size
		for aspect in aspects:
			label[aspect] = 1
			self.weight[aspect] += 1
		return label
	
	def _generate_binary_label_sentiment(self, aspects, sentiment_values):
		label = [-1] * self.aspect_size
		count = 0
		for i, aspect in enumerate(aspects):
			if aspect == 1:
				sv = sentiment_values[count]
				label[i] = sv
				count += 1
		return label