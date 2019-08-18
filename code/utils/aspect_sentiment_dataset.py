import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class AspectSentiData(Dataset):
	def __init__(self, df, test=False, aspect_size=10, tag_size=3, padding_value=13690):
		self.sentiment_hash = {-1:0, 0:1, 1:2}
		self.df = df
		self.labels = []
		self.data = []
		self.aspect_size = aspect_size
		self.tag_size = tag_size
		self.clause_len = []
		self.article_len = []
		max_clause_len = 0
		self.weight = [0]*10
		self.idxs = []
		self.test = test
		for i, idx, content in zip(df['content_seg_idx'].index, df['content_id'], df['content_seg_idx']):
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
			self.clause_len.append(','.join(l))
			if test is False:
				aspect = [int(aidx) for aidx in df['aspect_idx'][i].split(',')]
				label = [int(ll) for ll in df['sentiment_value'][i].split(',')]
				binary_label = self._generate_binary_label(aspect, label)
				self.labels.append(binary_label)
			self.article_len.append(len(l))
			self.data.append(clause_idx)
			self.idxs.append(idx)

		for i in range(len(self.data)):
			for j in range(len(self.data[i])):
				clause_len = len(self.data[i][j])
				if clause_len < max_clause_len:
					self.data[i][j].extend([padding_value for _ in range(max_clause_len-clause_len)])
			self.data[i] = torch.LongTensor(self.data[i])
		self.data = pad_sequence(self.data, batch_first=True, padding_value=padding_value)

		# total_samples = sum(self.weight)
		# self.weight = [total_samples/w for w in self.weight]
		# min_w = min(self.weight)

	def __getitem__(self, index):
		if self.test:
			return self.idxs[index], self.data[index], self.clause_len[index], self.article_len[index]
		else:
			return self.idxs[index], self.data[index], torch.Tensor(self.labels[index]), self.clause_len[index], self.article_len[index]

	def __len__(self):
		return len(self.data)

	def _generate_binary_label(self, aspects, labels):
		label = [[0 for _ in range(self.tag_size)] for _ in range(self.aspect_size)]
		for i, aspect in enumerate(aspects):
			sv = self.sentiment_hash[int(labels[i])]
			label[aspect][sv] = 1
			self.weight[aspect] += 1
		binary_label = []
		for l in label:
			binary_label.extend(l)
		return binary_label