import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class TextDataBinary(Dataset):
	def __init__(self, df, target_labels, test=False, aspect_size=10, padding_value=13690):
		self.df = df
		self.target_labels = target_labels
		self.sentiment_hash = {}
		if 0 in target_labels:
			self.sentiment_hash[0] = 1
			neg_label = 1 if 1 in target_labels else 0
			self.sentiment_hash[neg_label] = 0
		else:
			self.sentiment_hash = {
				-1 : 0, 1 : 1
			}
		self.aspects = []
		self.data = []
		self.aspect_size = aspect_size
		self.clause_len = []
		self.article_len = []
		self.labels = []
		max_clause_len = 0
		self.weight = [0]*10
		self.test = test
		self.idxs = []
		for i, idx, content, aspect in zip(df['content_seg_idx'].index, df['content_id'], df['content_seg_idx'], df['aspect_idx']):
			clauses = content.split('\t')
			clause_idx = []
			l = []
			for clause in clauses:
				zzz = [int(idx) for idx in clause.split(' ')]
				zzz_len = len(zzz)
				l.append(str(zzz_len))
				clause_idx.append(zzz)
				if zzz_len > max_clause_len:
					max_clause_len = zzz_len
			asp = [int(aidx) for aidx in aspect.split(',')]
			asp = self._generate_binary_label(asp)
			self.clause_len.append(','.join(l))
			self.article_len.append(len(l))
			self.data.append(clause_idx)
			self.aspects.append(asp)
			self.idxs.append(idx)
			if test is False:
				senti_value = df['sentiment_value'][i]
				senti_v = [int(z) for z in senti_value.split(',')]
				label = self._generate_binary_label_sentiment(asp, senti_v)
				self.labels.append(label)
				if sum(asp) == 0:
					self.clause_len.pop()
					self.article_len.pop()
					self.data.pop()
					self.aspects.pop()
					self.idxs.pop()
					self.labels.pop()

		for i in range(len(self.data)):
			for j in range(len(self.data[i])):
				clause_len = len(self.data[i][j])
				if clause_len < max_clause_len:
					self.data[i][j].extend([padding_value for _ in range(max_clause_len-clause_len)])
			self.data[i] = torch.LongTensor(self.data[i])
		self.data = pad_sequence(self.data, batch_first=True, padding_value=padding_value)

	def __getitem__(self, index):
		if self.test:
			return self.idxs[index], self.data[index], torch.Tensor(self.aspects[index]), self.clause_len[index], self.article_len[index]
		else:
			return self.idxs[index], self.data[index], torch.Tensor(self.aspects[index]), torch.LongTensor(self.labels[index]), self.clause_len[index], self.article_len[index]

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
				if sentiment_values[count] not in self.sentiment_hash:
					aspects[i] = 2
				else:
					label[i] = self.sentiment_hash[sentiment_values[count]]
				count += 1
		return label