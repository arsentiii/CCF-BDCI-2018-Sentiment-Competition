import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class AspectData(Dataset):
	def __init__(self, df, aspect_size=10, padding_value=13690, \
		pos_padding_value=30, char_padding_value=2848, \
		max_word_char_len=4, word_pinyin_padding_value=3677, \
		char_pinyin_padding_value=65):
		self.df = df
		self.labels = []
		self.data = []
		self.data_one_level = []
		self.pos = []
		self.pos_one_level = []
		self.aspect_size = aspect_size
		self.clause_len = []
		self.article_len = []
		self.clause_chars = []
		self.char = []
		self.char_len = []
		self.word_len = []
		self.word_pinyin = []
		self.word_pinyin_one_level = []
		max_clause_len = 0
		max_char_len = 0
		max_word_len = 0
		self.weight = [0]*10
		max_lll = 50
		# max_lll2 = 100
		for content, aspect, content_pos, chars, clause_chars,\
		content_one_level, content_pos_one_Level, char_pinyin, word_pinyin, pinyin_one_level\
		 in zip(df['content_seg_idx'], df['aspect_idx'], \
		 	df['content_pos_idx'], df['char_idx'], \
		 	df['clause_char_idx'], df['content_seg_idx_one_level'], \
		 	df['content_pos_idx_one_level'], df['content_pinyin_idx'], \
			df['content_word_pinyin_idx'], df['content_word_pinyin_idx_one_level']):
			clauses = content.split('\t')
			poss = content_pos.split('\t')
			clauses_char = clause_chars.split('\t')
			clause_word_pinyin = word_pinyin.split('\t')
			clause_idx = []
			pos_idx = []
			clause_char_idx = []
			clause_word_pinyin_idx = []
			l = []
			for clause, pos, cc, cwp in zip(clauses, poss, clauses_char, clause_word_pinyin):
				zzz = [int(idx) for idx in clause.split(' ')][:max_lll]
				poszzz = [int(idx) for idx in pos.split(' ')][:max_lll]
				cczzz = [[int(iii) for iii in idx.split('|')][:max_word_char_len]
                                    for idx in cc.split(' ')][:max_lll]
				cwpzzz = [int(idx) for idx in cwp.split(' ')][:max_lll]
				for b in range(len(cczzz)):
					if len(cczzz[b]) < max_word_char_len:
						cczzz[b].extend([char_padding_value for _ in range(max_word_char_len-len(cczzz[b]))])
				zzz_len = len(zzz)
				l.append(str(zzz_len))
				clause_idx.append(zzz)
				pos_idx.append(poszzz)
				clause_word_pinyin_idx.append(cwpzzz)
				clause_char_idx.append(cczzz)
				if zzz_len > max_clause_len:
					max_clause_len = zzz_len
			self.clause_len.append(','.join(l))
			self.article_len.append(len(l))
			char_seg = [int(idx) for idx in chars.split(' ')]
			char_len = len(char_seg)
			if char_len > max_char_len:
				max_char_len = char_len
			word_one_level = [int(idx) for idx in content_one_level.split(' ')]
			pos_one_Level = [int(idx) for idx in content_pos_one_Level.split(' ')]
			pinyin_one_level = [int(idx) for idx in pinyin_one_level.split(' ')]
			word_len = len(word_one_level)
			if word_len > max_word_len:
				max_word_len = word_len
			self.data_one_level.append(word_one_level)
			self.pos_one_level.append(pos_one_Level)
			self.word_pinyin_one_level.append((pinyin_one_level))
			self.word_len.append(word_len)
			self.clause_chars.append(clause_char_idx)
			self.char.append(char_seg)
			self.char_len.append(char_len)
			aspects = [int(idx) for idx in aspect.split(',')]
			label = self._generate_binary_label(aspects)
			self.data.append(clause_idx)
			self.pos.append(pos_idx)
			self.word_pinyin.append(clause_word_pinyin_idx)
			self.labels.append(label)

		for i in range(len(self.data)):
			for j in range(len(self.data[i])):
				clause_len = len(self.data[i][j])
				if clause_len < max_clause_len:
					self.data[i][j].extend([padding_value for _ in range(max_clause_len-clause_len)])
					self.pos[i][j].extend([pos_padding_value for _ in range(max_clause_len-clause_len)])
					self.clause_chars[i][j].extend([[char_padding_value]*max_word_char_len for _ in range(max_clause_len-clause_len)])
					self.word_pinyin[i][j].extend([word_pinyin_padding_value for _ in range(max_clause_len-clause_len)])
			if len(self.char[i]) < max_char_len:
				self.char[i].extend([char_padding_value for _ in range(max_char_len-len(self.char[i]))])
			if self.word_len[i] < max_word_len:
				self.data_one_level[i].extend([padding_value for _ in range(max_word_len-self.word_len[i])])
				self.pos_one_level[i].extend([pos_padding_value for _ in range(max_word_len-self.word_len[i])])
				self.word_pinyin_one_level[i].extend([word_pinyin_padding_value for _ in range(max_word_len-self.word_len[i])])
			self.data[i] = torch.LongTensor(self.data[i])
			self.pos[i] = torch.LongTensor(self.pos[i])
			self.char[i] = torch.LongTensor(self.char[i])
			self.clause_chars[i] = torch.LongTensor(self.clause_chars[i])
			self.clause_chars[i] = torch.LongTensor(self.clause_chars[i])
			self.data_one_level[i] = torch.LongTensor(self.data_one_level[i])
			self.pos_one_level[i] = torch.LongTensor(self.pos_one_level[i])
			self.word_pinyin_one_level[i] = torch.LongTensor(self.word_pinyin_one_level[i])
			self.word_pinyin[i] = torch.LongTensor(self.word_pinyin[i])
		self.data = pad_sequence(self.data, batch_first=True, padding_value=padding_value)
		self.pos = pad_sequence(self.pos, batch_first=True, padding_value=pos_padding_value)
		self.char = pad_sequence(self.char, batch_first=True, padding_value=char_padding_value)
		self.clause_chars = pad_sequence(self.clause_chars, batch_first=True, padding_value=char_padding_value)
		self.data_one_level = pad_sequence(self.data_one_level, batch_first=True, padding_value=padding_value)
		self.pos_one_level = pad_sequence(self.pos_one_level, batch_first=True, padding_value=pos_padding_value)
		self.word_pinyin = pad_sequence(self.word_pinyin, batch_first=True, padding_value=word_pinyin_padding_value)
		self.word_pinyin_one_level = pad_sequence(self.word_pinyin_one_level, batch_first=True, padding_value=word_pinyin_padding_value)

	def __getitem__(self, index):
		return self.data[index], torch.Tensor(self.labels[index]), \
		self.clause_len[index], self.article_len[index], \
		self.pos[index], self.char[index], \
		self.char_len[index], self.clause_chars[index], \
		self.data_one_level[index], self.pos_one_level[index], \
		self.word_len[index], self.word_pinyin[index], self.word_pinyin_one_level[index]

	def __len__(self):
		return len(self.data)

	def _generate_binary_label(self, aspects):
		label = [0] * self.aspect_size
		for aspect in aspects:
			label[aspect] = 1
			self.weight[aspect] += 1
		return label
