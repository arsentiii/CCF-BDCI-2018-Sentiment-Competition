import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class TextDataSentence(Dataset):
	def __init__(self, df, test=False, aspect_size=10, padding_value=13690):
		self.sentiment_hash = {-1:0, 0:1, 1:2}
		self.df = df
		self.aspects = []
		self.data = []
		self.aspect_size = aspect_size
		self.article_len = []
		self.labels = []
		max_sentence_len = 0
		self.test = test
		self.idxs = []
		for i, idx, content, aspect in zip(df['content_seg_idx'].index, df['content_id'], df['content_seg_idx'], df['aspect_idx']):
			clauses = content.split('\t')
			sentence_idx = []
			for clause in clauses:
				zzz = [int(cidx) for cidx in clause.split(' ')]
				zzz_len = len(zzz)
				sentence_idx.extend(zzz)
			zzz_len = len(sentence_idx)
			if zzz_len > max_sentence_len:
				max_sentence_len = zzz_len
			asp = [int(aidx) for aidx in aspect.split(',')]
			asp = self._generate_binary_label(asp)
			if test is False:
				senti_value = df['sentiment_value'][i]
				senti_v = [self.sentiment_hash[int(z)] for z in senti_value.split(',')]
				label = self._generate_binary_label_sentiment(asp, senti_v)
				self.labels.append(label)
			self.article_len.append(zzz_len)
			self.data.append(sentence_idx)
			self.aspects.append(asp)
			self.idxs.append(idx)
		for i in range(len(self.data)):
			self.data[i] = torch.LongTensor(self.data[i])
		self.data = pad_sequence(self.data, batch_first=True, padding_value=padding_value)

	def __getitem__(self, index):
		if self.test:
			return self.idxs[index], self.data[index], torch.Tensor(self.aspects[index]), self.article_len[index]
		else:
			return self.idxs[index], self.data[index], torch.Tensor(self.aspects[index]), torch.LongTensor(self.labels[index]), self.article_len[index]

	def __len__(self):
		return len(self.data)

	def _generate_binary_label(self, aspects):
		label = [0] * self.aspect_size
		for aspect in aspects:
			label[aspect] = 1
		return label
	
	def _generate_binary_label_sentiment(self, aspects, sentiment_values):
		label = [-1] * self.aspect_size
		count = 0
		for i, aspect in enumerate(aspects):
			if aspect == 1:
				label[i] = sentiment_values[count]
				count += 1
		return label