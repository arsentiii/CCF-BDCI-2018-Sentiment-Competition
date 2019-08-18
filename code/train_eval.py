import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MultiLabelSoftMarginLoss, BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import gc

aspect_hash = {'动力': 0, '价格': 1, '油耗': 2, '操控': 3,
               '舒适性': 4, '配置': 5, '安全性': 6, '内饰': 7, '外观': 8, '空间': 9}
reverse_aspect_hash = {0: '动力', 1: '价格', 2: '油耗', 3: '操控',
                       4: '舒适性', 5: '配置', 6: '安全性', 7: '内饰', 8: '外观', 9: '空间'}



def merge_clause(text, text_len, clause_len,
                 aspect, label, pos, chars,
                 char_len, clause_chars, word_pinyin, word_pinyin_one_level,
                 word_pinyin_len, clause_chars2, clause_char_len,
                 text_one_level, word_len):
	x = []
	clause_lens = []
	new_aspect = []
	single_new_aspect = []
	new_text_len = []
	new_label = []
	idxs = []
	position = []
	all_clause_len = []
	new_pos = []
	new_chars = []
	new_char_len = []
	new_clause_chars = []
	new_word_pinyin = []
	new_word_pinyin_one_level = []
	new_word_pinyin_len = []
	new_clause_chars2 = []
	new_clause_char_len = []
	new_text_one_level = []
	new_word_len = []
	for cl in clause_len:
		all_clause_len.extend([int(z) for z in cl.split(',')])
	for ccl in clause_char_len:
		new_clause_char_len.extend([int(z) for z in ccl.split(',')])
	# max_clause_len = max(max(all_clause_len), 3)
	# max_clause_char_len = max(max(new_clause_char_len), 3)
	max_clause_len = max(all_clause_len)
	max_clause_char_len = max(new_clause_char_len)
	new_clause_char_len = []
	max_char_len = char_len.max().item()
	max_py_len = word_pinyin_len.max().item()
	max_word_len = word_len.max().item()
	chars = chars[:, :max_char_len]
	pos = pos[:, :, :max_clause_len]
	word_pinyin = word_pinyin[:, :, :max_clause_len]
	text = text[:, :, :max_clause_len]
	clause_chars2 = clause_chars2[:, :, :max_clause_char_len]
	if label is not None:
		label = label[:, :max_clause_len]
	clause_chars = clause_chars[:, :, :max_clause_len, :]
	word_pinyin_one_level = word_pinyin_one_level[:, :max_py_len]
	text_one_level = text_one_level[:, :max_word_len]
	for index in range(text.size(0)):
		tl = text_len[index].unsqueeze(0)
		chl = char_len[index].unsqueeze(0)
		pyl = word_pinyin_len[index].unsqueeze(0)
		tol = word_len[index].unsqueeze(0)
		if label is not None:
			l = label[index:index+1, :]
		clauses = text[index, :tl, :]
		clause_char2 = clause_chars2[index, :tl, :]
		clause_char = clause_chars[index, :tl, :, :]
		cur_aspect = aspect[index:index+1, :]
		cpw = word_pinyin[index, :tl, :]
		cur_pos = pos[index, :tl, :]
		cur_chars = chars[index:index+1, :]
		cur_pinyin = word_pinyin_one_level[index:index+1, :]
		cur_text_one = text_one_level[index:index+1, :]
		sum_a = cur_aspect.sum().item()
		cl = [int(z) for z in clause_len[index].split(',')]
		ccl = [int(z) for z in clause_char_len[index].split(',')]
		if sum_a > 1:
			for j in range(cur_aspect.size(1)):
				if cur_aspect[0, j].item() == 1:
					x.append(clauses)
					new_word_pinyin_one_level.append(cur_pinyin)
					new_word_pinyin.append(cpw)
					new_clause_chars.append(clause_char)
					new_text_one_level.append(cur_text_one)
					new_a = torch.zeros_like(cur_aspect)
					new_a[0, j] = 1
					new_aspect.append(new_a)
					new_pos.append(cur_pos)
					new_chars.append(cur_chars)
					new_clause_chars2.append(clause_char2)
					for k in range(tl):
						single_new_aspect.append(new_a)
					clause_lens.extend(cl)
					new_text_len.append(tl)
					new_char_len.append(chl)
					new_clause_char_len.extend(ccl)
					new_word_pinyin_len.append(pyl)
					new_word_len.append(tol)
					idxs.append(index)
					position.append(j)
					if label is not None:
						new_label.append(l[0, j].unsqueeze(0))
		else:
			for k in range(tl):
				single_new_aspect.append(cur_aspect)
			x.append(clauses)
			new_word_pinyin_one_level.append(cur_pinyin)
			new_word_pinyin.append(cpw)
			new_clause_chars.append(clause_char)
			new_clause_chars2.append(clause_char2)
			new_text_one_level.append(cur_text_one)
			new_pos.append(cur_pos)
			new_chars.append(cur_chars)
			new_aspect.append(cur_aspect)
			clause_lens.extend(cl)
			new_char_len.append(chl)
			new_text_len.append(tl)
			new_clause_char_len.extend(ccl)
			new_word_pinyin_len.append(pyl)
			new_word_len.append(tol)
			for j in range(cur_aspect.size(1)):
				if cur_aspect[0, j].item() == 1:
					if label is not None:
						new_label.append(l[0, j].unsqueeze(0))
					position.append(j)
			idxs.append(index)
	return x, clause_lens, [new_aspect, single_new_aspect], \
            new_text_len, new_label, idxs, position, \
            new_pos, new_chars, new_char_len, \
            new_clause_chars, new_word_pinyin, new_word_pinyin_one_level, \
            new_word_pinyin_len, new_clause_chars2, new_clause_char_len, \
            new_text_one_level, new_word_len


def merge_clause2(aspect, label, chars, char_len, word, word_len):
	new_aspect = []
	new_label = []
	idxs = []
	position = []
	new_chars = []
	new_char_len = []
	new_word = []
	new_word_len = []
	for index in range(word.size(0)):
		chl = char_len[index].unsqueeze(0)
		wl = word_len[index].unsqueeze(0)
		if label is not None:
			l = label[index:index+1, :]
		cur_aspect = aspect[index:index+1, :]
		cur_chars = chars[index:index+1, :]
		cur_word = word[index:index+1, :]
		sum_a = cur_aspect.sum().item()
		if sum_a > 1:
			for j in range(cur_aspect.size(1)):
				if cur_aspect[0, j].item() == 1:
					new_a = torch.zeros_like(cur_aspect)
					new_a[0, j] = 1
					new_aspect.append(new_a)
					new_chars.append(cur_chars)
					new_char_len.append(chl)
					new_word.append(cur_word)
					new_word_len.append(wl)
					idxs.append(index)
					position.append(j)
					if label is not None:
						new_label.append(l[0, j].unsqueeze(0))
		else:
			new_chars.append(cur_chars)
			new_aspect.append(cur_aspect)
			new_word.append(cur_word)
			new_word_len.append(wl)
			new_char_len.append(chl)
			for j in range(cur_aspect.size(1)):
				if cur_aspect[0, j].item() == 1:
					if label is not None:
						new_label.append(l[0, j].unsqueeze(0))
					position.append(j)
			idxs.append(index)
	return new_aspect, new_label, idxs, position, \
            new_chars, new_char_len, new_word, new_word_len


def merge_clause_aspect(text, text_len, clause_len, pos, clause_chars, word_pinyin):
	x = []
	new_pos = []
	clause_lens = []
	new_clasue_chars = []
	new_word_pinyin = []
	for cl in clause_len:
		clause_lens.extend([int(z) for z in cl.split(',')])
	mcl = max(clause_lens)
	for index in range(text.size(0)):
		tl = text_len[index]
		clauses = text[index, :tl, :mcl]
		clause_char = clause_chars[index, :tl, :mcl, :]
		word_py = word_pinyin[index, :tl, :mcl]
		x.append(clauses)
		new_pos.append(pos[index, :tl, :mcl])
		new_clasue_chars.append(clause_char)
		new_word_pinyin.append(word_py)
	return x, clause_lens, new_pos, new_clasue_chars, new_word_pinyin


def train_aspect(args, model, model_name,
                 optimizer, train_aspect_loader, val_aspect_loader,
                 device, model_path, fold, thes):
	ignored_params = list(map(id, model.HUARN.word_embedding.parameters()))
	ignored_params.extend(
		list(map(id, model.HUARN_pinyin.word_embedding.parameters())))

	ignored_params.extend(
		list(map(id, model.word_nn.word_embedding.parameters())))
	base_params = filter(lambda p: id(p) not in ignored_params,
                      model.parameters())
	ignored_params = filter(lambda p: id(p) in ignored_params, model.parameters())
	criterion = BCEWithLogitsLoss().to(device)
	best_score = 0.0
	best_pred = None
	best_result = None
	last_score = 0.0
	optimizer = optim.Adam([
            {'params': base_params},
            {'params': ignored_params, 'lr': 0}
        ], lr=1e-3, weight_decay=1e-4)
	lr = 1e-3
	for epoch in range(args.epochs):
		if epoch == args.fine_tune_epochs:
			optimizer.param_groups[1]['lr'] = 2e-4
		for i, (text, label, clause_len, \
			text_len, pos, chars, char_len, \
			clause_chars, text_one_level, \
			pos_one_level, word_len, word_pinyin, word_pinyin_one_level)\
			in enumerate(train_aspect_loader):
			model.train()
			text_len = text_len.to(device)
			label = label.to(device)
			max_char_len = char_len.max().item()
			chars = chars[:, :max_char_len]
			chars = chars.to(device)
			char_len = char_len.to(device)
			max_word_len = word_len.max().item()
			text_one_level = text_one_level[:, :max_word_len].to(device)
			word_len = word_len.to(device)
			x, clause_lens, new_pos, new_clause_chars, new_word_pinyin \
			= merge_clause_aspect(text, text_len, clause_len, pos, clause_chars, word_pinyin)
			x = torch.cat(x, dim=0).to(device)
			new_pos = torch.cat(new_pos, dim=0).to(device)
			new_clause_chars = torch.cat(new_clause_chars, dim=0).to(device)
			new_word_pinyin = torch.cat(new_word_pinyin, dim=0).to(device)
			logits, _ = model(
				x, clause_lens, text_len, new_pos,
				chars, char_len, new_clause_chars,
				new_word_pinyin, text_one_level, word_len
				# text_one_level, pos_one_level, word_len
			)
			loss = criterion(logits, label)
			model.zero_grad()
			optimizer.zero_grad()
			loss.backward()
			# torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
			optimizer.step()
			# for name, parameters in model.named_parameters():
			# 	print(name, parameters.grad)
			# continue
			print('Model %s\t\t Epoch %d\t\t Iter %d\t\t Loss %0.4f' %
			      (model_name, epoch, i, loss.item()))
			if i == 16 or i == 32:
				result, pred = eval_aspect(
					args, model, model_name, val_aspect_loader, device, 'val', fold, thes)
				mirco_fs = result[0][-1]
				if mirco_fs > best_score:
					best_score = mirco_fs
					best_result = result
					best_pred = pred
					torch.save(model.state_dict(), model_path)
				if mirco_fs < last_score and epoch > args.fine_tune_epochs:
					lr /= 2.0
					optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
				last_score = mirco_fs
			torch.cuda.empty_cache()
	return best_result, best_pred


def train_sentiment(args, model, model_name,
                    optimizer, train_text_loader, val_text_loader,
                    device, model_path, fold, pred_val_aspect=None, 
					aspect_size=10, score=0.0, fine_tune=False, acc_step=2):

	best_score1 = score
	best_result1 = None
	best_score2 = 0.0
	best_result2 = None
	best_score3 = 0.0
	best_result3 = None
	ignored_params = list(map(id, model.HUARN.word_embedding.parameters()))
	ignored_params.extend(
		list(map(id, model.aspect_embedding.parameters())))
	ignored_params.extend(
		list(map(id, model.HUARN.self_word_embedding.parameters())))
	ignored_params.extend(
		list(map(id, model.HUARN.aspect_embedding.parameters())))
	ignored_params.extend(
		list(map(id, model.HUARN_pinyin_rep.word_embedding.parameters())))
	ignored_params.extend(
		list(map(id, model.HUARN_pinyin_rep.self_word_embedding.parameters())))
	ignored_params.extend(
		list(map(id, model.HUARN_pinyin_rep.aspect_embedding.parameters())))
	# ignored_params.extend(
	# 	list(map(id, model.HUARN_char_rep.aspect_embedding.parameters())))
	ignored_params.extend(
		list(map(id, model.char_encoder.aspect_embedding.parameters())))
	ignored_params.extend(
		list(map(id, model.word_encoder.word_embedding.parameters())))
	ignored_params.extend(
		list(map(id, model.word_encoder.self_word_embedding.parameters())))
	ignored_params.extend(
		list(map(id, model.word_encoder.aspect_embedding.parameters())))
	ignored_params = []
	base_params = filter(lambda p: id(p) not in ignored_params,
                      model.parameters())
	ignored_params = filter(lambda p: id(p) in ignored_params, model.parameters())
	criterion = CrossEntropyLoss().to(device)
	optimizer = optim.Adam([
            {'params': base_params},
            {'params': ignored_params, 'lr': 0}
        ], lr=1e-3, weight_decay=1e-4)
	total_loss = 0.0
	lr = 1e-3
	last_score = score
	criterion = CrossEntropyLoss().to(device)
	# epochs = args.epochs if not fine_tune else args.fine_tune_epochs
	for epoch in range(args.epochs_senti):
		model.train()
		for i, (idx, text, aspect, label, clause_len,
                        text_len, pos, chars, char_len, clause_chars,
                        text_one_level, pos_one_level, word_len, word_pinyin,
                        word_pinyin_one_level, word_pinyin_len,
                        clause_chars2, clause_char_len) \
                        in enumerate(train_text_loader):
			model.train()
			if epoch == 2:
				optimizer.param_groups[1]['lr'] = 2e-4
			text_len = text_len.to(device)
			label = label.to(device)
			aspect = aspect.to(device)
			max_char_len = char_len.max().item()
			chars = chars[:, :max_char_len]
			chars = chars.to(device)

			x, clause_lens, new_aspect, new_text_len, \
                            new_label, idxs, position, new_pos, \
                            new_chars, new_char_len, new_clause_chars, new_word_pinyin, \
                            new_word_pinyin_one_level, new_word_pinyin_len, \
                            new_clause_chars2, new_clause_char_len, \
                            new_text_one_level, new_word_len \
                            = merge_clause(text, text_len, clause_len,
                                           aspect, label, pos, chars,
                                           char_len, clause_chars, word_pinyin,
                                           word_pinyin_one_level, word_pinyin_len,
                                           clause_chars2, clause_char_len,
                                           text_one_level, word_len)
			x = torch.cat(x, dim=0).to(device)
			new_a = torch.cat(new_aspect[0], dim=0).float().to(device)
			single_new_a = torch.cat(new_aspect[1], dim=0).float().to(device)
			new_text_len = torch.cat(new_text_len, dim=0).to(device)
			new_label = torch.cat(new_label, dim=0).to(device)
			# new_pos = torch.cat(new_pos, dim=0).to(device)
			new_chars = torch.cat(new_chars, dim=0).to(device)
			new_char_len = torch.cat(new_char_len, dim=0).to(device)
			new_clause_chars = torch.cat(new_clause_chars, dim=0).to(device)
			new_word_pinyin = torch.cat(new_word_pinyin, dim=0).to(device)
			# new_word_pinyin_one_level = torch.cat(new_word_pinyin_one_level, dim=0).to(device)
			new_word_pinyin_len = torch.cat(new_word_pinyin_len, dim=0).to(device)
			# new_clause_chars2 = torch.cat(new_clause_chars2, dim=0).to(device)
			# aspect_label = torch.LongTensor(position, device=device)
			new_text_one_level = torch.cat(new_text_one_level, dim=0).to(device)
			new_word_len = torch.cat(new_word_len, dim=0).to(device)
			logits = model(x, [new_a, single_new_a],
                            clause_lens, new_text_len, idxs,
                            position, new_pos, new_chars,
                            new_char_len, new_clause_chars,
                            new_word_pinyin, new_word_pinyin_one_level,
                            new_word_pinyin_len, new_text_one_level, new_word_len)

			loss = criterion(logits, new_label) / acc_step
			loss.backward()
			total_loss += loss.item()
			if (i + 1) % acc_step == 0:
				optimizer.step()
				model.zero_grad()
				optimizer.zero_grad()
				print('Model %s\t\t Epoch %d\t\t Iter %d\t\t Loss %0.10f' %
				      (model_name, epoch, i//2, total_loss))
				total_loss = 0.0
			# torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
			
			del x, new_a, single_new_a, \
                            clause_lens, new_text_len, idxs, \
                            position, new_pos, new_chars, \
                            new_char_len, new_clause_chars, \
                            new_word_pinyin, new_word_pinyin_one_level, \
                            new_word_pinyin_len
			if (i + 1) % acc_step == 0 and (i//acc_step == 16 or i//acc_step == 32):
				result = eval_sentiment(args, model, model_name, val_text_loader,
				                        device, 'val', fold, pred_val_aspect, aspect_size)
				mirco_fs = result[0][-1]
				if mirco_fs > best_score1:
					best_score1 = mirco_fs
					best_result1 = result
					torch.save(model.state_dict(), model_path)
				if fine_tune:
					if mirco_fs < last_score:
						optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
					last_score = mirco_fs
				if mirco_fs < last_score and epoch > 2:
					lr /= 2.0
					# optimizer.param_groups[0]['lr'] = lr
					# optimizer.param_groups[1]['lr'] = lr
					optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
				# last_score = mirco_fs
			gc.collect()
			torch.cuda.empty_cache()
	return best_result1


def eval_aspect(args, model, model_name, aspect_loader, device, eval_type, fold, thes):
	model.eval()
	all_pred = []
	all_label = []
	if 'val' in eval_type:
		tp = fp = fn = 0
	for i, (text, label, clause_len,
         text_len, pos, chars, char_len,
         clause_chars, text_one_level,
         pos_one_level, word_len, word_pinyin, word_pinyin_one_level)\
                in enumerate(aspect_loader):
		text_len = text_len.to(device)
		label = label.long().cpu().detach().numpy()
		max_char_len = char_len.max().item()
		chars = chars[:, :max_char_len]
		chars = chars.to(device)
		char_len = char_len.to(device)
		max_word_len = word_len.max().item()
		text_one_level = text_one_level[:, :max_word_len].to(device)
		word_len = word_len.to(device)
		x, clause_lens, new_pos, new_clause_chars, new_word_pinyin\
		= merge_clause_aspect(text, text_len, clause_len, pos, clause_chars, word_pinyin)
		x = torch.cat(x, dim=0).to(device)
		new_pos = torch.cat(new_pos, dim=0).to(device)
		new_clause_chars = torch.cat(new_clause_chars, dim=0).to(device)
		new_word_pinyin = torch.cat(new_word_pinyin, dim=0).to(device)
		_, prob = model(
			x, clause_lens, text_len, new_pos,
			chars, char_len, new_clause_chars,
			new_word_pinyin, text_one_level, word_len
		)
		prob = prob.cpu().detach().numpy()
		if 'val' in eval_type:
			pred = (prob > thes) * 1
			flag = (pred + label)
			tp += np.sum((flag == 2) * 1, axis=0)
			fp += np.sum((flag + pred == 2) * 1, axis=0)
			fn += np.sum((flag + label == 2) * 1, axis=0)
		all_pred.append(prob)
		all_label.append(label)
		del x, clause_lens, text_len, new_pos, chars, char_len, new_clause_chars, \
                    new_word_pinyin, text_one_level, word_len
		gc.collect()
		torch.cuda.empty_cache()
	# print(tp, fp, fn)
	final_pred = np.concatenate(all_pred, axis=0)
	final_label = np.concatenate(all_label, axis=0)

	if 'val' in eval_type:
		mAP = average_precision_score(label, prob)
		pbase = tp + fp
		rbase = tp + fn
		pbase[pbase == 0] = 1
		rbase[rbase == 0] = 1
		p = tp / pbase
		r = tp / rbase
		fbase = p + r
		fbase[fbase == 0] = 1
		f = 2 * p * r / fbase

		gtp = np.sum(tp)
		gfp = np.sum(fp)
		gfn = np.sum(fn)
		gp = (gtp / (gtp + gfp)) if gtp > 0 else 0
		gr = (gtp / (gtp + gfn)) if gtp > 0 else 0
		gf = (2 * gp * gr / (gp + gr)) if 2 * gp * gr > 0 else 0

		lp = np.mean(p)
		lr = np.mean(r)
		lf = np.mean(f)

		for i in range(10):
			aspect = reverse_aspect_hash[i]
			print('Fold %d\t Aspect %s\t P %0.4f R %0.4f F %0.4f' %
			      (fold, aspect, p[i].item(), r[i].item(), f[i].item()))
		print('Fold %d\t Marco\t\t P %0.4f R %0.4f F %0.4f' % (fold, lp, lr, lf))
		print('Fold %d\t Micro\t\t P %0.4f R %0.4f F %0.4f' % (fold, gp, gr, gf))
		print('Fold %d\t Model %s mAP %0.4f' % (fold, model_name, mAP))
		print('-------------------------------------------')

	if eval_type == 'val':
		return ([gp, gr, gf], [lp, lr, lf], mAP), final_pred
	else:

		return final_pred, final_label


def eval_sentiment(
	args, model, model_name, text_loader, device,
	eval_type, fold, pred_aspects=None, aspect_size=10):

	model.eval()
	label_hash = {0: -1, 1: 0, 2: 1}
	all_prob = []
	all_index = []
	all_position = []
	ptr = 0
	all_idx = []
	val_loss = 0.0
	if eval_type == 'val':
		tp = [0] * 3
		fp = [0] * 3
		fn = [0] * 3
		p = [0] * 3
		r = [0] * 3
		f = [0] * 3
		task_tp = task_fp = task_fn = 0
	for i, data in enumerate(text_loader):
		if eval_type == 'val':
			(idx, text, aspect, label, clause_len,
                            text_len, pos, chars, char_len, clause_chars,
                            text_one_level, pos_one_level, word_len,
                            word_pinyin, word_pinyin_one_level,
                            word_pinyin_len, clause_chars2, clause_char_len) = data
			label = label.to(device)
			aspect = aspect.to(device)
		else:
			(idx, text, aspect, clause_len,
                            text_len, pos, chars, char_len, clause_chars,
                            text_one_level, pos_one_level, word_len,
                            word_pinyin, word_pinyin_one_level,
                            word_pinyin_len, clause_chars2, clause_char_len) = data
			all_idx.extend(idx)
			label = None
		batch_size = aspect.size(0)
		pred_aspect = pred_aspects[ptr:ptr+batch_size, :].to(device)
		ptr += batch_size
		if eval_type == 'test':
			aspect = pred_aspect

		text_len = text_len.to(device)
		max_char_len = char_len.max().item()
		chars = chars[:, :max_char_len]
		chars = chars.to(device)
		x, clause_lens, new_aspect, new_text_len, \
                    new_label, idxs, position, new_pos, \
                    new_chars, new_char_len, new_clause_chars, new_word_pinyin, \
                    new_word_pinyin_one_level, new_word_pinyin_len, \
                    new_clause_chars2, new_clause_char_len, \
                    new_text_one_level, new_word_len \
                    = merge_clause(text, text_len, clause_len,
                                   aspect, label, pos, chars,
                                   char_len, clause_chars, word_pinyin,
                                   word_pinyin_one_level, word_pinyin_len,
                                   clause_chars2, clause_char_len,
                                   text_one_level, word_len)
		x = torch.cat(x, dim=0).to(device)
		new_a = torch.cat(new_aspect[0], dim=0).float().to(device)
		single_new_a = torch.cat(new_aspect[1], dim=0).float().to(device)
		new_text_len = torch.cat(new_text_len, dim=0).to(device)
		# new_pos = torch.cat(new_pos, dim=0).to(device)
		new_chars = torch.cat(new_chars, dim=0).to(device)
		new_char_len = torch.cat(new_char_len, dim=0).to(device)
		new_clause_chars = torch.cat(new_clause_chars, dim=0).to(device)
		new_word_pinyin = torch.cat(new_word_pinyin, dim=0).to(device)
		# new_word_pinyin_one_level = torch.cat(new_word_pinyin_one_level, dim=0).to(device)
		new_word_pinyin_len = torch.cat(new_word_pinyin_len, dim=0).to(device)
		# new_clause_chars2 = torch.cat(new_clause_chars2, dim=0).to(device)
		new_text_one_level = torch.cat(new_text_one_level, dim=0).to(device)
		new_word_len = torch.cat(new_word_len, dim=0).to(device)
		if len(new_label) > 0:
			new_label = torch.cat(new_label, dim=0).to(device)
		logits = model(x, [new_a, single_new_a],
                 clause_lens, new_text_len, idxs, position,
                 new_pos, new_chars, new_char_len, new_clause_chars,
                 new_word_pinyin, new_word_pinyin_one_level,
                 new_word_pinyin_len, new_text_one_level, new_word_len)

		# max_word_len = word_len.max().item()
		# text_one_level = text_one_level[:, :max_word_len]
		# new_aspect, new_label, idxs, position, \
		# new_chars, new_char_len, new_word, new_word_len \
		# = merge_clause2(aspect, label, chars, char_len, text_one_level, word_len)
		# x = torch.cat(new_word, dim=0).to(device)
		# x_len = torch.cat(new_word_len, dim=0).to(device)
		# new_char = torch.cat(new_chars, dim=0).to(device)
		# new_char_len = torch.cat(new_char_len, dim=0).to(device)
		# new_aspect = torch.cat(new_aspect, dim=0).to(device)
		# if len(new_label) > 0:
		# 	new_label = torch.cat(new_label, dim=0).to(device)
		# logits = model(x, x_len, new_char, new_char_len, new_aspect)
		prob = torch.softmax(logits, dim=-1)
		del x, clause_lens, new_aspect, new_text_len, \
                    new_pos, new_chars, new_char_len, \
                    new_clause_chars, new_word_pinyin, \
                    new_word_pinyin_one_level, new_word_pinyin_len, \
                    new_clause_chars2, new_clause_char_len
		gc.collect()
		torch.cuda.empty_cache()
		if eval_type == 'test':
			all_prob.append(prob.cpu().detach().numpy())
			all_index.append(idxs)
			all_position.append(position)
		if eval_type == 'val':
			# val_loss += criterion(logits, new_label).item()
			pred = prob.topk(1)[1].squeeze(1)
			tp_fp_fn(pred, new_label, tp, fp, fn)
			'''
			Val for Predicted Aspect
			'''
			x, clause_lens, new_aspect, \
                            new_text_len, new_label, idxs, \
                            position, new_pos, new_chars, \
                            new_char_len, new_clause_chars, \
                            new_word_pinyin, new_word_pinyin_one_level, \
                            new_word_pinyin_len, new_clause_chars2, new_clause_char_len, \
                            new_text_one_level, new_word_len \
                            = merge_clause(text, text_len, clause_len,
                                           pred_aspect, label, pos, chars,
                                           char_len, clause_chars, word_pinyin,
                                           word_pinyin_one_level, word_pinyin_len,
                                           clause_chars2, clause_char_len,
                                           text_one_level, word_len)
			x = torch.cat(x, dim=0).to(device)
			new_a = torch.cat(new_aspect[0], dim=0).to(device)
			single_new_a = torch.cat(new_aspect[1], dim=0).to(device)
			new_text_len = torch.cat(new_text_len, dim=0).to(device)
			# new_pos = torch.cat(new_pos, dim=0).to(device)
			new_chars = torch.cat(new_chars, dim=0).to(device)
			new_char_len = torch.cat(new_char_len, dim=0).to(device)
			new_clause_chars = torch.cat(new_clause_chars, dim=0).to(device)
			new_word_pinyin = torch.cat(new_word_pinyin, dim=0).to(device)
			new_word_pinyin_len = torch.cat(new_word_pinyin_len, dim=0).to(device)
			# new_clause_chars2 = torch.cat(new_clause_chars2, dim=0).to(device)
			new_text_one_level = torch.cat(new_text_one_level, dim=0).to(device)
			new_word_len = torch.cat(new_word_len, dim=0).to(device)
			logits = model(x, [new_a, single_new_a],
                            clause_lens, new_text_len, idxs, position,
                            new_pos, new_chars, new_char_len,
                            new_clause_chars, new_word_pinyin,
                            new_word_pinyin_one_level, new_word_pinyin_len, new_text_one_level, new_word_len)
			prob = torch.softmax(logits, dim=-1)
			pred_aspect_pred = prob.topk(1)[1].squeeze(1)
			sub_task_tp, sub_task_fp, sub_task_fn = task_tp_fp_fn(
				pred_aspect_pred, label, aspect, pred_aspect, idxs, position)
			task_tp += sub_task_tp
			task_fp += sub_task_fp
			task_fn += sub_task_fn
			# del x, clause_lens, new_aspect, new_text_len, new_label, idxs, position
			gc.collect()
			torch.cuda.empty_cache()
			# exit(0)
	if eval_type == 'val':
		for i in range(len(tp)):
			pbase = max(1, (tp[i] + fp[i]))
			rbase = max(1, (tp[i] + fn[i]))
			p[i] = tp[i] / pbase
			r[i] = tp[i] / rbase
			fbase = max(1, (p[i] + r[i]))
			f[i] = 2 * p[i] * r[i] / fbase
			print('Label %d\t P %0.4f\t R %0.4f\t F%0.4f' % (i, p[i], r[i], f[i]))
		gtp = sum(tp)
		gfp = sum(fp)
		gfn = sum(fn)
		gp = gtp / (gtp + gfp)
		gr = gtp / (gtp + gfn)
		gf = 2 * gp * gr / (gp + gr)
		lp = sum(p) / len(p)
		lr = sum(r) / len(r)
		lf = sum(f) / len(f)
		task_p = task_tp / (task_tp + task_fp)
		task_r = task_tp / (task_tp + task_fn)
		task_f = 2 * task_p * task_r / (task_p + task_r)
		val_loss /= float(i+1)
		print('Micro\t P %0.4f\t R %0.4f\t F%0.4f' % (gp, gr, gf))
		print('Marco\t P %0.4f\t R %0.4f\t F%0.4f' % (lp, lr, lf))
		print('Task\t P %0.4f\t R %0.4f\t F%0.4f' % (task_p, task_r, task_f))
		print('Val Loss %0.10f' % val_loss)
		result = form_final_result(all_prob, all_index, all_position)
		return ([gp, gr, gf], [lp, lr, lf], [p, r, f], [task_p, task_r, task_f], val_loss)
	else:
		result = form_final_result(all_prob, all_index, all_position)
		return result, all_idx


def form_final_result(all_prob, all_index, all_position, aspect_size=10, tag_size=3):
	size = sum([len(set(idxs)) for idxs in all_index])
	# final_result = torch.zeros(size, aspect_size, tag_size)
	final_result = np.zeros(shape=(size, aspect_size, tag_size))
	cur_index = 0
	for idxs, pos, prob in zip(all_index, all_position, all_prob):
		for i, idx in enumerate(idxs):
			final_result[cur_index+idx, pos[i], :] = prob[i, :]
		cur_index += len(set(idxs))
	return final_result


def tp_fp_fn(pred, label, tp, fp, fn):
	for p, l in zip(pred, label):
		p = p.item()
		l = l.item()
		if p == l:
			tp[p] += 1
		elif p != l:
			fp[p] += 1
			fn[l] += 1


def extract_corr_label(pred_aspect, label, idxs, position):
	matrix_size = len(set(idxs))
	aspect_size = pred_aspect.size(1)
	matrix = [[-2 for _ in range(aspect_size)] for _ in range(matrix_size)]
	for i, idx in enumerate(idxs):
		matrix[idx][position[i]] = label[i].item()
		if pred_aspect[idx, position[i]].item() == 0:
			print(pred_aspect[idx], idx, position[i])
	return matrix


def task_tp_fp_fn(pred_aspect_pred, label, aspect, pred_aspect, idxs, position):
	# for i in range(pred_aspect.size(0)):
	# 	print(pred_aspect[i])
	# print(position)
	# exit(0)
	tp = fp = fn = 0
	batch_size = len(set(idxs))
	aspect_size = aspect.size(1)
	pred = extract_corr_label(pred_aspect, pred_aspect_pred, idxs, position)
	for i in range(batch_size):
		p = pred[i]
		l = label[i]
		a = aspect[i]
		pa = pred_aspect[i]

		diff = a.sum().item() - pa.sum().item()
		if diff < 0:
			fp -= diff
		elif diff > 0:
			fn += diff
		for j in range(aspect_size):
			if a[j] == 1:
				if a[j] == pa[j] and l[j] == p[j]:
					tp += 1
				else:
					fp += 1
	return tp, fp, fn


def extract_idxs(idxs, aspect):
	multi_idxs = []
	for i in range(len(idxs)):
		idx = idxs[i]
		a = aspect[i]
		multi_idxs.extend([idx for _ in range(a.sum().item())])
	return multi_idxs


def to_csv(aspect, label, idxs, path):
	aspect_hash = {
		0: '动力', 1: '价格', 2: '油耗',
		3: '操控', 4: '舒适性', 5: '配置',
		6: '安全性', 7: '内饰', 8: '外观', 9: '空间'
	}
	label_hash = {
		0: -1, 1: 0, 2: 1
	}
	df = pd.DataFrame()
	size = aspect.size(1)
	list_aspect = []
	list_label = []
	list_idx = []
	for i in range(len(idxs)):
		a = aspect[i]
		l = [int(zzz) for zzz in label[i]]
		idx = idxs[i]
		for j in range(size):
			aspect_tag = a[j].item()
			if aspect_tag == 1:
				list_aspect.append(aspect_hash[j])
				list_label.append(label_hash[l[j]])
				list_idx.append(idx)
	df['content_id'] = list_idx
	df['subject'] = list_aspect
	df['sentiment_value'] = list_label
	df['sentiment_word'] = [''] * len(list_idx)
	df.to_csv(path, index=None)
