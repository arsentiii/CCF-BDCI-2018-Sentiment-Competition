import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


class TextData(Dataset):
	def __init__(self, df, test=False, aspect_size=10,
              padding_value=13690, pos_padding_value=30,
              char_padding_value=2848, max_word_char_len=4,
              word_pinyin_padding_value=3677, char_pinyin_padding_value=65):
		self.sentiment_hash = {-1: 0, 0: 1, 1: 2}
		self.df = df
		self.aspects = []
		self.data = []
		self.data_one_level = []
		self.pos = []
		self.pos_one_level = []
		self.aspect_size = aspect_size
		self.clause_len = []
		self.clause_char_len = []
		self.article_len = []
		self.word_len = []
		self.labels = []
		self.char = []
		self.char_len = []
		self.clause_chars = []
		self.clause_chars2 = []
		max_clause_len = 0
		max_clause_char_len = 0
		max_char_len = 0
		max_word_len = 0
		max_pinyin_len = 0
		self.weight = [0]*10
		self.test = test
		self.idxs = []
		self.word_pinyin = []
		self.char_pinyin = []
		self.word_pinyin_one_level = []
		self.word_pinyin_len = []
		max_lll = 50
		max_lll2 = 2000
		for i, uidx, content, aspect, content_pos, chars, clause_chars, \
                    content_one_level, content_pos_one_Level, \
                    char_pinyin, word_pinyin, word_pinyin_one_level \
                    in zip(df['content_seg_idx'].index, df['content_id'],
                           df['content_seg_idx'], df['aspect_idx'], df['content_pos_idx'],
                           df['char_idx'], df['clause_char_idx'], df['content_seg_idx_one_level'],
                           df['content_pos_idx_one_level'], df['content_pinyin_idx'],
                           df['content_word_pinyin_idx'], df['content_word_pinyin_idx_one_level']):
			clauses = content.split('\t')
			poss = content_pos.split('\t')
			clause_pinyin = word_pinyin.split('\t')
			clauses_char = clause_chars.split('\t')
			clause_idx = []
			pos_idx = []
			l = []
			l2 = []
			clause_char_idx = []
			clause_char_idx2 = []
			clause_word_pinyin_idx = []
			for clause, pos, cc, cwp \
                                in zip(clauses, poss, clauses_char, clause_pinyin):
				# zzz = [int(idx) for idx in clause.split(' ')][:20]
				# poszzz = [int(idx) for idx in pos.split(' ')][:20]
				zzz = [int(idx) for idx in clause.split(' ')][:max_lll]
				poszzz = [int(idx) for idx in pos.split(' ')][:max_lll]
				cczzz = []
				cczzz2 = []
				for idx in cc.split(' '):
					cczzzz = []
					for iii in idx.split('|'):
						ccidx = int(iii)
						cczzzz.append(ccidx)
						cczzz2.append(ccidx)
					# cczzz.append(cczzzz[:20])
				cczzz = [[int(iii) if iii != '' else char_padding_value for iii in idx.split(
					'|')][:max_word_char_len] for idx in cc.split(' ')][:max_lll]
				cwpzzz = [int(idx) for idx in cwp.split(' ')][:max_lll]
				# cwpzzz = [int(idx) for idx in cwp.split(' ')][:20]
				# cczzz2 = cczzz2[:30]
				# cczzz = cczzz[:20]
				for b in range(len(cczzz)):
					if len(cczzz[b]) < max_word_char_len:
						cczzz[b].extend(
							[char_padding_value for _ in range(max_word_char_len-len(cczzz[b]))])
				zzz_len = len(zzz)
				zzz2_len = len(cczzz2)
				l.append(str(zzz_len))
				l2.append(str(zzz2_len))
				clause_idx.append(zzz)
				pos_idx.append(poszzz)
				clause_char_idx.append(cczzz)
				clause_char_idx2.append(cczzz2)
				clause_word_pinyin_idx.append(cwpzzz)
				if zzz_len > max_clause_len:
					max_clause_len = zzz_len
				if zzz2_len > max_clause_char_len:
					max_clause_char_len = zzz2_len
			aspect = str(aspect)
			asp = [int(aidx) for aidx in aspect.split(',')]
			asp = self._generate_binary_label(asp)
			if test is False:
				senti_value = str(df['sentiment_value'][i])
				senti_v = [self.sentiment_hash[int(z)] for z in senti_value.split(',')]
				label = self._generate_binary_label_sentiment(asp, senti_v)
				self.labels.append(label)
			self.clause_len.append(','.join(l))
			self.clause_char_len.append(','.join(l2))
			char_seg = [int(idx) for idx in chars.split(' ')][:max_lll2]
			char_len = len(char_seg)
			if char_len > max_char_len:
				max_char_len = char_len
			word_one_level = [int(idx) for idx in content_one_level.split(' ')][:max_lll2]
			pos_one_Level = [int(idx) for idx in content_pos_one_Level.split(' ')][:max_lll2]
			word_len = len(word_one_level)
			if word_len > max_word_len:
				max_word_len = word_len
			word_pinyin_one_level = [int(idx)
                            for idx in word_pinyin_one_level.split(' ')]
			word_pinyin_one_level_len = len(word_pinyin_one_level)
			if word_pinyin_one_level_len > max_pinyin_len:
				max_pinyin_len = word_pinyin_one_level_len
			self.data_one_level.append(word_one_level)
			self.pos_one_level.append(pos_one_Level)
			self.word_len.append(word_len)
			self.clause_chars.append(clause_char_idx)
			self.clause_chars2.append(clause_char_idx2)
			self.word_pinyin.append(clause_word_pinyin_idx)
			self.word_pinyin_one_level.append(word_pinyin_one_level)
			self.word_pinyin_len.append(word_pinyin_one_level_len)
			self.char.append(char_seg)
			self.char_len.append(char_len)
			self.article_len.append(len(l))
			self.data.append(clause_idx)
			self.aspects.append(asp)
			self.pos.append(pos_idx)
			self.idxs.append(uidx)

		# max_clause_len = 20
		# max_clause_char_len = 30
		# # max_char_len = 30
		# max_word_len = 20
		# max_pinyin_len = 20
		for i in range(len(self.data)):
			for j in range(len(self.data[i])):
				clause_len = len(self.data[i][j])
				if clause_len < max_clause_len:
					self.data[i][j].extend(
						[padding_value for _ in range(max_clause_len-clause_len)])
					self.pos[i][j].extend(
						[pos_padding_value for _ in range(max_clause_len-clause_len)])
					self.clause_chars[i][j].extend(
						[[char_padding_value]*max_word_char_len for _ in range(max_clause_len-clause_len)])
					self.word_pinyin[i][j].extend(
						[word_pinyin_padding_value for _ in range(max_clause_len-clause_len)])
				clause_char_len = len(self.clause_chars2[i][j])
				if clause_char_len < max_clause_char_len:
					self.clause_chars2[i][j].extend(
						[char_padding_value for _ in range(max_clause_char_len-clause_char_len)])
			if len(self.char[i]) < max_char_len:
				self.char[i].extend(
					[char_padding_value for _ in range(max_char_len-len(self.char[i]))])
			if self.word_len[i] < max_word_len:
				self.data_one_level[i].extend(
					[padding_value for _ in range(max_word_len-self.word_len[i])])
				self.pos_one_level[i].extend(
					[pos_padding_value for _ in range(max_word_len-self.word_len[i])])
			if self.word_pinyin_len[i] < max_pinyin_len:
				self.word_pinyin_one_level[i].extend(
					word_pinyin_padding_value for _ in range(max_pinyin_len-self.word_pinyin_len[i]))
			self.data[i] = torch.LongTensor(self.data[i])
			self.pos[i] = torch.LongTensor(self.pos[i])
			self.char[i] = torch.LongTensor(self.char[i])
			self.clause_chars[i] = torch.LongTensor(self.clause_chars[i])
			self.data_one_level[i] = torch.LongTensor(self.data_one_level[i])
			self.pos_one_level[i] = torch.LongTensor(self.pos_one_level[i])
			self.word_pinyin[i] = torch.LongTensor(self.word_pinyin[i])
			self.word_pinyin_one_level[i] = torch.LongTensor(
				self.word_pinyin_one_level[i])
			self.clause_chars2[i] = torch.LongTensor(self.clause_chars2[i])
		self.data = pad_sequence(self.data, batch_first=True,
		                         padding_value=padding_value)
		self.pos = pad_sequence(self.pos, batch_first=True,
		                        padding_value=pos_padding_value)
		self.char = pad_sequence(self.char, batch_first=True,
		                         padding_value=char_padding_value)
		self.clause_chars = pad_sequence(
			self.clause_chars, batch_first=True, padding_value=char_padding_value)
		self.data_one_level = pad_sequence(
			self.data_one_level, batch_first=True, padding_value=padding_value)
		self.pos_one_level = pad_sequence(
			self.pos_one_level, batch_first=True, padding_value=pos_padding_value)
		self.word_pinyin = pad_sequence(
			self.word_pinyin, batch_first=True, padding_value=word_pinyin_padding_value)
		self.word_pinyin_one_level = pad_sequence(
			self.word_pinyin_one_level, batch_first=True, padding_value=word_pinyin_padding_value)
		self.clause_chars2 = pad_sequence(
			self.clause_chars2, batch_first=True, padding_value=char_padding_value)

	def __getitem__(self, index):
		if self.test:
			return self.idxs[index], self.data[index], \
                            torch.Tensor(self.aspects[index]), self.clause_len[index], \
                            self.article_len[index], self.pos[index], self.char[index], \
                            self.char_len[index], self.clause_chars[index], \
                            self.data_one_level[index], self.pos_one_level[index], self.word_len[index], \
                            self.word_pinyin[index], self.word_pinyin_one_level[index], \
                            self.word_pinyin_len[index], self.clause_chars2[index], self.clause_char_len[index]
		else:
			return self.idxs[index], self.data[index], \
                            torch.Tensor(self.aspects[index]), torch.LongTensor(self.labels[index]), \
                            self.clause_len[index], self.article_len[index], \
                            self.pos[index], self.char[index], \
                            self.char_len[index], self.clause_chars[index], \
                            self.data_one_level[index], self.pos_one_level[index], self.word_len[index], \
                            self.word_pinyin[index], self.word_pinyin_one_level[index], \
                            self.word_pinyin_len[index], self.clause_chars2[index], self.clause_char_len[index]

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
