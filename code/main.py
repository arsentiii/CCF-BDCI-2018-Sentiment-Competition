import argparse
import torch
import numpy as np, pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import statistics
import random
from sklearn.model_selection import KFold, ShuffleSplit
from utils import *
from train_eval import *
from model import *
import gc
import os

random.seed(6)
np.random.seed(6)
torch.manual_seed(6)
if torch.cuda.is_available():
	torch.cuda.manual_seed(6)
	torch.cuda.manual_seed_all(6)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Car Comment Sentiment Analysis')
parser.add_argument('--seed', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--params', type=bool, default=str)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--epochs-senti', type=int, default=8)
parser.add_argument('--fine-tune-epochs', type=int, default=2)
parser.add_argument('--fine-tune-epochs-senti', type=int, default=3)

def check_zero(pred, prob):
	for i in range(pred.shape[0]):
		if np.sum(pred[i, :]) == 0:
			top = np.argmax(prob[i, :])
			pred[i, top] = 1
	return pred

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)

def find_optimal_thes(pred, label):
	optim_thes = [0] * 10
	for i in range(10):
		iprob, ilabel = pred[:, i], label[:, i]
		thess = [0.5 + 0.01 * j for j in range(36)]
		best_f = 0.0
		for thes in thess:
			thes = round(thes, 2)
			ipred = (iprob > thes) * 1
			flag = (ipred + ilabel)
			tp = np.sum((flag == 2) * 1, axis=0)
			fp = np.sum((flag + ipred == 2) * 1, axis=0)
			fn = np.sum((flag + ilabel == 2) * 1, axis=0)
			fbase = 2 * tp + fp + fn
			f = 2 * tp / fbase if fbase > 0 else 0
			if f >= best_f:
				optim_thes[i] = thes
				best_f = f
	return optim_thes

def get_idx(bools):
	idxs = []
	for i, b in enumerate(bools):
		if b:idxs.append(i)
	return idxs

def _init_fn(worker_id):
	# np.random.seed(6 + worker_id)
	return torch.initial_seed() + worker_id

def main(args, train, test, device, n_splits=5, thes=0.5):
	embedding = {}
	for et in ['word2vec', 'glove', 'fasttext']:
		aspect_embedding_path = '../../data/nn_data/' + et + '_aspect_embedding2.npy'
		word_embedding_path = '../../data/nn_data/' + et + '_word_embedding2.npy'
		word_pinyin_embedding_path = '../../data/nn_data/' + et + '_py_embedding.300.npy'
		if not os.path.exists(aspect_embedding_path) or \
		not os.path.exists(word_embedding_path) or \
		not os.path.exists(word_pinyin_embedding_path):
			print('error')
			exit(0)
		embedding[et] = {}
		embedding[et]['aspect_embedding'] = np.load(aspect_embedding_path)
		embedding[et]['word_embedding'] = np.load(word_embedding_path)
		embedding[et]['word_pinyin_embedding'] = np.load(word_pinyin_embedding_path)
	padding_value = embedding['word2vec']['word_embedding'].shape[0] - 1
	pos_padding_value = 28
	pos_vocab_size = 29
	# char_padding_value = 2374
	# char_vocab_size = 2375
	# char_padding_value = 2144
	char_padding_value = 2310
	char_vocab_size = char_padding_value + 1
	word_pinyin_vocab_size = embedding['word2vec']['word_pinyin_embedding'].shape[0]
	word_pinyin_padding_value = word_pinyin_vocab_size - 1
	char_pinyin_vocab_size = 66
	char_pinyin_padding_value = char_pinyin_vocab_size - 1
	'''
	Train and Val for Aspect Nets
	'''
	model_fs = {
		'AspectNet' : 0.0, 
		'SequenceAspectNet' : 0.0, 
		'RethinkSequenceNet' : 0.0		
	}
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)	
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	word_embedding = embedding['word2vec']['word_embedding']
	word_pinyin_embedding = embedding['word2vec']['word_pinyin_embedding']
	aspect_embedding = embedding['word2vec']['aspect_embedding']
	char_nn_params = [300, 150, 300, None, char_vocab_size, device, 1]
	pinyin_nn_params = (300, 150, 300, None, char_vocab_size, device, 1)
	word_nn_params = (300, 150, 300, None, char_vocab_size, device, 1, 10, True, word_embedding, pos_vocab_size)
	kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
	names = ['AspectNet', 'SequenceAspectNet', 'RethinkSequenceNet']
	names = ['SequenceAspectNet']
	for et in ['word2vec', 'glove', 'fasttext']:
		set_seed(args)
		word_embedding = embedding[et]['word_embedding']
		word_pinyin_embedding = embedding[et]['word_pinyin_embedding']
		char_nn_params = [300, 150, 300, None, char_vocab_size, device, 1]
		pinyin_nn_params = (300, 150, 300, None, char_vocab_size, device, 1)
		word_nn_params = (300, 150, 300, None, char_vocab_size, device, 1, 10, True, word_embedding, pos_vocab_size)
		kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
		for i, (train_index, val_index) in enumerate(kf.split(train)):
			x_train, x_val = train.iloc[train_index], train.iloc[val_index]
			train_aspect_data = AspectData(
				x_train, padding_value=padding_value, pos_padding_value=pos_padding_value, 
				char_padding_value=char_padding_value, word_pinyin_padding_value=word_pinyin_padding_value,
				char_pinyin_padding_value=char_pinyin_padding_value)
			val_aspect_data = AspectData(
				x_val, padding_value=padding_value, pos_padding_value=pos_padding_value, 
				char_padding_value=char_padding_value, word_pinyin_padding_value=word_pinyin_padding_value,
				char_pinyin_padding_value=char_pinyin_padding_value)
			train_aspect_loader = DataLoader(
				train_aspect_data, batch_size=args.batch_size, drop_last=True, shuffle=True)
			val_aspect_loader = DataLoader(
				val_aspect_data, batch_size=args.batch_size, shuffle=False)
			aspect_net = AspectNet(300, [150, 150], 10, word_embedding, word_pinyin_embedding, pos_vocab_size, char_nn_params, device).to(device)
			seq_aspect_net = SequenceAspectNet(300, [150, 150], 10, word_embedding, word_pinyin_embedding, pos_vocab_size, char_nn_params, device).to(device)
			rethink_aspect_net = RethinkAspectNet(300, [150, 150], 10, 10, word_embedding, word_pinyin_embedding, pos_vocab_size, char_nn_params, device).to(device)
			aspect_models = {
				'AspectNet' : aspect_net, 
				'SequenceAspectNet' : seq_aspect_net, 
				'RethinkSequenceNet' : rethink_aspect_net
			}
			for model_name in names:
				model_path = '../../model/aspect_model_2nd/' + et + '_' + \
				model_name + '_' + str(i) + '3embedding'
				model = aspect_models[model_name]
				optimizer = None
				# optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
				# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-6)
				result, pred = train_aspect(
					args, model, model_name, optimizer, train_aspect_loader, 
					val_aspect_loader, device, model_path, i, thes)
				mirco_fs = result[0][-1]
				marco_fs = result[1][-1]
				mAP = result[-1]
				model_fs[model_name] += mirco_fs / n_splits
				print('Model %s\t Fold %d\t Mirco F1 %0.4f\t Marco F1 %0.4f\t mAP %0.4f' % (model_name, i, mirco_fs, marco_fs, mAP))
				print('Next Model----------------------')
				torch.cuda.empty_cache()
				# return
			print('Next Fold----------------------')
		torch.cuda.empty_cache()
		for model_name in model_fs:
			print('Embedding %s Model %s\t Avg Mirco F1 %0.4f' % (et, model_name, model_fs[model_name]))
	exit(0)
	
	n_splits_val_pred = []
	n_splits_val_prob = []
	n_splits_val_label = []
	fold_test_pred = 0.0
	for i, (train_index, val_index) in enumerate(kf.split(train)):
		aspect_net = AspectNet(300, [150, 150], 10, word_embedding, word_pinyin_embedding, pos_vocab_size, char_nn_params, device).to(device)
		seq_aspect_net = SequenceAspectNet(300, [150, 150], 10, word_embedding, word_pinyin_embedding, pos_vocab_size, char_nn_params, device).to(device)
		rethink_aspect_net = RethinkAspectNet(300, [150, 150], 10, 10, word_embedding, word_pinyin_embedding, pos_vocab_size, char_nn_params, device).to(device)
		aspect_models = {
			'AspectNet' : aspect_net, 
			'SequenceAspectNet' : seq_aspect_net, 
			'RethinkSequenceNet' : rethink_aspect_net
		}
		x_val = train.iloc[val_index]
		val_aspect_data = AspectData(x_val, padding_value=padding_value, pos_padding_value=pos_padding_value, char_padding_value=char_padding_value)
		val_aspect_loader = DataLoader(
			val_aspect_data, batch_size=args.batch_size, shuffle=False)
			# , num_workers=4, worker_init_fn=_init_fn)
		test_aspect_data = AspectData(test, padding_value=padding_value, pos_padding_value=pos_padding_value, char_padding_value=char_padding_value)
		test_aspect_loader = DataLoader(
			test_aspect_data, batch_size=args.batch_size, shuffle=False)
			# , num_workers=4, worker_init_fn=_init_fn)
		fold_val_pred = 0.0
		for model_name in names:
			ets = ['word2vec', 'glove', 'fasttext']
			for et in ets:
				print(i, et, model_name)
				# model_path = '../../model/aspect_model_with_punc_pos_word_char_pinyin/other7_' + model_name + '_' + str(i) + '.model_remove_low_fre_3' 
				# model_path = '../../model/aspect_new_embedding_remove_stopword/remove_stopword_' + model_name + '_' + str(i) + '.model_remove_low_fre_3_5fold'
				# model_path = '../../model/aspect_model_2nd/' + et + \
	   #                          model_name + '_' + str(i)
				model_path = '../../model/aspect_model_2nd/' + et + '_' + \
				model_name + '_' + str(i) + '3embedding'
				model = aspect_models[model_name]
				state_dict = torch.load(model_path)
				model.load_state_dict(state_dict)
				del state_dict
				gc.collect()
				torch.cuda.empty_cache()
				final_val_pred, final_val_label = eval_aspect(args, model, model_name, val_aspect_loader, device, 'test', i, thes)
				final_test_pred, _ = eval_aspect(args, model, model_name, test_aspect_loader, device, 'test', i, thes)
				fold_val_pred += final_val_pred / (len(aspect_models) * len(ets))
				fold_test_pred += final_test_pred / (len(aspect_models) * n_splits * len(ets))
		n_splits_val_prob.append(fold_val_pred)
		n_splits_val_label.append(final_val_label)
		del aspect_net, seq_aspect_net, rethink_aspect_net, val_aspect_data, test_aspect_data
		gc.collect()
		torch.cuda.empty_cache()
	all_fold_prob = np.concatenate(n_splits_val_prob, axis=0)
	all_fold_label = np.concatenate(n_splits_val_label, axis=0)
	optim_thes = find_optimal_thes(all_fold_prob, all_fold_label)
	print('Optimal Thes', optim_thes)
	for i in range(n_splits):
		fold_val_pred = n_splits_val_prob[i]
		thes = np.zeros(fold_val_pred.shape)
		for j, t in enumerate(optim_thes):
			thes[:, j] = t
		fold_val_pred_aspect = (fold_val_pred > thes) * 1
		fold_val_pred_aspect = check_zero(fold_val_pred_aspect, fold_val_pred)
		n_splits_val_pred.append(fold_val_pred_aspect)
	test_thes = np.zeros(fold_test_pred.shape)
	for j, t in enumerate(optim_thes):
		test_thes[:, j] = t
	test_pred_aspect = (fold_test_pred > test_thes) * 1
	check_zero(test_pred_aspect, fold_test_pred)
	torch.cuda.empty_cache()

	np.save('../../tmp/2nd/test_3_embedding.tmp', test_pred_aspect)
	for i in range(n_splits):
		np.save('../../tmp/2nd/val_3_embedding' + str(i) + '.tmp', n_splits_val_pred[i])
	exit(0)

	
	# n_splits_val_pred = []
	# for i in range(n_splits):
	# 	n_splits_val_pred.append(torch.load('../../tmp/other7_n_splits_val_pred_with_punc_pos_word_char_pinyin_remove_low_fre_3' + str(i) +'.tmp'))\

	final_mirco_fs1 = final_marco_fs1 = final_task_fs1 = 0.0
	final_mirco_fs2 = final_marco_fs2 = final_task_fs2 = 0.0
	final_mirco_fs3 = final_marco_fs3 = final_task_fs3 = 0.0
	acc_step = 2
	# kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
	ets = ['word2vec', 'glove', 'fasttext']
	mirco_f = {}
	macro_f = {}
	task_f = {}
	avg_mirco_f = {}
	avg_macro_f = {}
	avg_task_f = {}
	for et in ets:
		mirco_f[et] = []
		macro_f[et] = []
		task_f[et] = []
		set_seed(args)
		for i, (train_index, val_index) in enumerate(kf.split(train)):
			word_embedding = embedding[et]['word_embedding']
			word_pinyin_embedding = embedding[et]['word_pinyin_embedding']
			aspect_embedding = embedding[et]['aspect_embedding']
			char_nn_params = [300, 150, 300, None, char_vocab_size, device, 1]
			pinyin_nn_params = (300, 150, 300, None, char_vocab_size, device, 1)
			word_nn_params = (300, 150, 300, None, char_vocab_size, device, 1, 10, True, word_embedding, pos_vocab_size)
			x_train, x_val = train.iloc[train_index], train.iloc[val_index]
			train_text_data = TextData(x_train, test=False, padding_value=padding_value, \
			pos_padding_value=pos_padding_value, \
			char_padding_value=char_padding_value, \
			word_pinyin_padding_value=word_pinyin_padding_value)
			val_text_data = TextData(x_val, test=False, padding_value=padding_value, \
			pos_padding_value=pos_padding_value, \
			char_padding_value=char_padding_value, \
			word_pinyin_padding_value=word_pinyin_padding_value)
			train_text_loader = DataLoader(
				train_text_data, batch_size=args.batch_size//acc_step, drop_last=True, shuffle=True)
			val_text_loader = DataLoader(
				val_text_data, batch_size=args.batch_size//2, shuffle=False)
			pred_val_aspect = torch.Tensor(
				np.load('../../tmp/2nd/val_3_embedding' + str(i) + '.tmp.npy'))
			model_name = 'SentimenNet5' + '_' + et + '_highway'
			model_path = '../../model/sentiment_model_2nd/' + model_name + '_' + str(i) + '.model' 
			model = SentimentNet(300, [150, 150], 10, 3, aspect_embedding, word_embedding, word_pinyin_embedding, pos_vocab_size, char_nn_params, pinyin_nn_params, device).to(device)		
			optimizer = None
			result1 = train_sentiment(
				args, model, model_name, optimizer, train_text_loader, 
				val_text_loader, device, model_path, i, pred_val_aspect, acc_step=acc_step)
			mirco_fs1 = result1[0][-1]
			print('Fold %d Best Mirco F1 Score %0.4f' % (i, mirco_fs1))
			del optimizer
			gc.collect()
			torch.cuda.empty_cache()
			# mirco_fs1 = 0.0
			# fine_tune_model_name = 'Fine_Tune_SentimenNet5'
			# fine_tune_model_path = '../../model/sentiment_model/' + fine_tune_model_name + '_' + str(i) + '.model' 
			# model.load_state_dict(torch.load(model_path))
			# model.HUARN.word_embedding.weight.requires_grad = True
			# model.HUARN.self_word_embedding.weight.requires_grad = True
			# model.HUARN.aspect_embedding.weight.requires_grad = True
			# model.HUARN_pinyin_rep.word_embedding.weight.requires_grad = True
			# model.HUARN_pinyin_rep.self_word_embedding.weight.requires_grad = True
			# model.HUARN_pinyin_rep.aspect_embedding.weight.requires_grad = True
			# model.HUARN_char_rep.aspect_embedding.weight.requires_grad = True
			# model.char_encoder.aspect_embedding.weight.requires_grad = True
			# model.word_encoder.word_embedding.weight.requires_grad = True
			# model.word_encoder.self_word_embedding.weight.requires_grad = True
			# model.word_encoder.aspect_embedding.weight.requires_grad = True
			# fine_tune_optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
			# result = train_sentiment(
			# 	args, model, fine_tune_model_name, 
			# 	fine_tune_optimizer, train_text_loader, val_text_loader, 
			# 	device, fine_tune_model_path, i, pred_val_aspect, 
			# 	score=mirco_fs1, fine_tune=True)
			# if result != None:
			# 	result1 = result
			mirco_fs1 = result1[0][-1]
			marco_fs1 = result1[1][-1]
			task_fs1 = result1[3][-1]
			val_loss = result1[-1]
			mirco_f[et].append(mirco_fs1)
			macro_f[et].append(marco_fs1)
			task_f[et].append(task_fs1)
			del model
			gc.collect()
			torch.cuda.empty_cache()
			print('Model %s\t Fold %d\t Mirco F1 %0.4f\t Marco F1 %0.4f\t Task F1 %0.4f' % (model_name, i, mirco_fs1, marco_fs1, task_fs1))
			print('Model %s\t Fold %d\t Val Loss %0.10f' % (model_name, i, val_loss))
		avg_mirco_f[et] = sum(mirco_f[et])/n_splits
		avg_macro_f[et] = sum(macro_f[et])/n_splits
		avg_task_f[et] = sum(task_f[et])/n_splits
		print('Avg Mirco F1 %0.4f\t Marco F1 %0.4f\t Task F1 %0.4f' % (avg_mirco_f[et], avg_macro_f[et], avg_task_f[et]))
		torch.cuda.empty_cache()
	total_mirco_f = 0.0
	total_macro_f = 0.0
	total_task_f = 0.0
	for et in ets:
		print('Embedding %s Mirco F1 %0.4f\t Macro F1 %0.4f\t Task F1 %0.4f' % (et, avg_mirco_f[et], avg_macro_f[et], avg_task_f[et]))
		total_mirco_f += avg_mirco_f[et]
		total_macro_f += avg_macro_f[et]
		total_task_f += avg_task_f[et]
	print('All Model Avg Mirco F1 %0.4f\t Macro F1 %0.4f\t Task F1 %0.4f' % (total_mirco_f/len(ets), total_macro_f/len(ets), total_task_f/len(ets)))
	exit(0)

	# test_pred_aspect = torch.Tensor(test_pred_aspect)
	test_pred_aspect = torch.Tensor(np.load('../../tmp/2nd/test_3_embedding.tmp.npy'))
	# test_pred_aspect = torch.load('../../tmp/new_model_test_pred.tmp')
	# test_pred_aspect = torch.load('../../tmp/other7_test_pred_aspect_with_punc_pos_word_char_pinyin_remove_low_fre_3.tmp')
	test_aspect_data = TextData(test, padding_value=padding_value, test=True, pos_padding_value=pos_padding_value, char_padding_value=char_padding_value)
	test_aspect_loader = DataLoader(
		test_aspect_data, batch_size=args.batch_size, shuffle=False)
		# , num_workers=4, worker_init_fn=_init_fn)
	for i in range(n_splits):
		ets = ['word2vec', 'glove', 'fasttext']
		pred = 0.0
		for et in ets:
			print(i, et)
			# model_name = 'SentimenNet5'
			# model_name = 'SentimenNet5' + '_Glove'
			model_name = 'SentimenNet5' + '_' + et + '_highway_conv'
			model_path = '../../model/sentiment_model_2nd/' + model_name + '_' + str(i) + '.model' 
			# model_path = '../../model/sentiment_model/' + model_name + '_' + str(i) + '.model' 
			# if not os.path.exists(model_path):
			# 	model_name = 'SentimenNet5'
			# 	model_path = '../../model/sentiment_model/' + model_name + '_' + str(i) + '.model'
			model = SentimentNet(300, [150, 150], 10, 3, aspect_embedding, word_embedding, word_pinyin_embedding, pos_vocab_size, char_nn_params, pinyin_nn_params, device).to(device)		
			# model = MILNET(300, [150, 150], 10, 3, aspect_embedding, word_embedding, word_pinyin_embedding, char_vocab_size, device).to(device)
			state_dict = torch.load(model_path)
			model.load_state_dict(state_dict)
			del state_dict
			gc.collect()
			torch.cuda.empty_cache()
			print(et, i)
			epred, all_idx = eval_sentiment(
				args, model, model_name, test_aspect_loader, device, 
				'test', i, test_pred_aspect, 10)
			pred += epred / len(ets)
		torch.save(pred, '../../result/' + 'single_sentiment_decoder_nn' + str(i) + '.bin')
	torch.save(all_idx, '../../result/single_sentiment_decoder_nn_idx.bin')
	
	all_idx = torch.load('../../result/single_sentiment_decoder_nn_idx.bin')
	n_split_pred = 0.0
	for i in range(n_splits):
		pred = torch.load('../../result/' + 'single_sentiment_decoder_nn' + str(i) + '.bin')
		n_split_pred += pred / n_splits
	n_split_tag = np.argmax(n_split_pred, axis=2)
	to_csv(test_pred_aspect, n_split_tag, all_idx, '../../result/single_sentiment_decoder_nn.csv')		

if __name__ == '__main__':
	args = parser.parse_args()
	# set_seed(args)	
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	# train = pd.read_csv('../../data/nn_data/train_with_idx_with_char_pos.csv', header=0)
	# test = pd.read_csv('../../data/nn_data/test_with_idx_with_char_pos.csv', header=0)
	
	train = pd.read_csv(
		'../../data/nn_data/train_with_idx_with_char_pos2.csv', header=0)
	test = pd.read_csv(
		'../../data/nn_data/test_with_idx_with_char_pos2.csv', header=0)
	main(args, train, test, device)
