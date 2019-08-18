import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .init_util import init_xavier_uniform


class HighWay(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(HighWay, self).__init__()
		self.tra = nn.Linear(input_size, hidden_size)
		self.wh = nn.Linear(hidden_size, hidden_size)
		self.wt = nn.Linear(hidden_size, 1)
		self.apply(init_xavier_uniform)

	def forward(self, x):
		x = torch.cat(x, -1)
		h = torch.relu(self.tra(x))
		t = torch.sigmoid(self.wt(h))
		z = t * torch.tanh(self.wh(h)) + (1. - t) * h
		return z
