import re
import sys
import os
import random
from random import shuffle
import codecs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import pandas as pd
import itertools
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

import unimodel as model 

def main(lr_in= 0.01, h_dim=150, do=0.60, wd_in=0.0001, lr_d=500,split=0.8, best_acc=[1]):
	random.seed(datetime.now()) 
	input_data, output_data, pmid_data = getData()
	vocab, vocab_rev, vectors = getWordVectors()

	data = []

	for i in range(len(input_data)):
		sent = sentenceToIndexs(input_data[i],vocab)
		sample = (sent,output_data[i],pmid_data[i])
		data.append(sample)

	train_data , test_data = trainTestSplit(data,split)

	shuffle(train_data)
	shuffle(test_data)

	test_in, test_out = paddInputs(test_data,vocab)
	test_in = autograd.Variable(torch.transpose(torch.LongTensor(test_in).cuda(),0,1))
	test_out = autograd.Variable(torch.LongTensor(test_out).cuda())
	test_pmid = [x[2] for x in test_data]

	test_running_loss = []
	val_running_loss = []
	train_running_loss = []
	crop_running_loss = []
	test_running_acc = []
	val_running_acc = []
	train_running_acc = []
	crop_running_acc = []
	test_running_pre = []
	val_running_pre = []
	train_running_pre = []
	crop_running_pre = []
	test_running_rec = []
	val_running_rec = []
	train_running_rec = []
	crop_running_rec = []

	my_model = model.RNN(len(vocab),len(vectors[0]),h_dim,2,dropout=do,word_vecs=vectors)
	#my_model = loadModel(my_model)
	my_model.cuda()

	input_cp, output_cp, pmid_cp = getTestData()

	cp = []

	for i in range(len(input_cp)):
		sent = sentenceToIndexs(input_cp[i],vocab)
		sample = (sent,output_cp[i],pmid_cp[i])
		cp.append(sample)

	crop_in, crop_out = paddInputs(cp,vocab)
	crop_in = autograd.Variable(torch.transpose(torch.LongTensor(crop_in).cuda(),0,1))
	crop_out = autograd.Variable(torch.LongTensor(crop_out).cuda())

	# scores = test(cp, vocab, my_model)
	# print ("ACC: " + str(scores[0]))
	# print ("PRE: " + str(scores[1]))
	# print ("REC: " + str(scores[2]))
	# exit()

	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(my_model.parameters(), lr=lr_in, weight_decay=wd_in)

	epochs = 150
	k_folds = 5
	
	folds_data = chunks(train_data,k_folds)
	
	for epoch in range(epochs):
		print ("\nEpoch: " + str(epoch+1))

		for i in range(len(folds_data)):

			val = []

			for j in range(len(folds_data[i])):
				val.append(folds_data[i][j])

			val_in, val_out = paddInputs(val,vocab)
			val_in = autograd.Variable(torch.transpose(torch.LongTensor(val_in).cuda(),0,1))
			val_out = autograd.Variable(torch.LongTensor(val_out).cuda())
			val_in.cuda()
			val_out.cuda()

			train = []
			for j in range(len(folds_data)):
				if j == i:
					continue
				train += folds_data[j]

			train_in, train_out = paddInputs(train,vocab)
			train_in = autograd.Variable(torch.transpose(torch.LongTensor(train_in).cuda(),0,1))
			train_out = autograd.Variable(torch.LongTensor(train_out).cuda())

			my_model.hidden = my_model.init_hidden(len(train))
			my_model.zero_grad()
			
			my_model.train()
			train_scores = my_model(train_in,training=True)
			
			train_loss = loss_function(train_scores, train_out)

			train_loss.backward(retain_graph=True)
			optimizer.step()
			
			optimizer=lr_decay(optimizer,epoch,lr_in,lr_d)			

			print("------------------------------")

			my_model.hidden = my_model.init_hidden(len(train))
				
			with torch.no_grad():
				train_scores_loss = my_model(train_in, training=True)
				train_scores_acc = my_model(train_in)

				train_loss = loss_function(train_scores_loss, train_out)
			train_running_loss.append(train_loss.data.cpu().numpy())

			train_pred = np.argmax(train_scores_acc.data.cpu().numpy(), axis=1)
			train_target = train_out.data.cpu().numpy()
			train_acc = np.sum(train_pred == train_target) / train_pred.shape[0]

			tp_count = 0
			fp_count = 0
			tn_count = 0
			fn_count = 0
			for i in range(len(train_pred)):
				if train_pred[i] == 1:
					if train_target[i] == 1:
						tp_count += 1
					else:
						fp_count += 1
				else:
					if train_target[i] == 0:
						tn_count += 1
					else:
						fn_count += 1
			train_running_pre.append(0 if tp_count <= 0 else tp_count/(tp_count+fp_count))
			train_running_rec.append(0 if tp_count <= 0 else tp_count/(tp_count+fn_count))
			train_running_acc.append(train_acc)

			print("------------------------------")

			my_model.hidden = my_model.init_hidden(len(val))

			my_model.eval()

			with torch.no_grad():
				val_scores_loss = my_model(val_in)
				val_scores_acc = my_model(val_in)

				val_loss = loss_function(val_scores_loss, val_out)
			val_running_loss.append(val_loss.data.cpu().numpy())

			val_pred = np.argmax(val_scores_acc.data.cpu().numpy(), axis=1)
			val_target = val_out.data.cpu().numpy()
			val_acc = np.sum(val_pred == val_target) / val_pred.shape[0]

			tp_count = 0
			fp_count = 0
			tn_count = 0
			fn_count = 0
			for i in range(len(val_pred)):
				if val_pred[i] == 1:
					if val_target[i] == 1:
						tp_count += 1
					else:
						fp_count += 1
				else:
					if val_target[i] == 0:
						tn_count += 1
					else:
						fn_count += 1
			val_running_pre.append(0 if tp_count <= 0 else tp_count/(tp_count+fp_count))
			val_running_rec.append(0 if tp_count <= 0 else tp_count/(tp_count+fn_count))
			val_running_acc.append(val_acc)

			print ("------------------------------")
			my_model.hidden = my_model.init_hidden(len(test_data))

			with torch.no_grad():
				test_scores_loss = my_model(test_in)
				test_scores_acc = my_model(test_in)

				test_loss = loss_function(test_scores_loss, test_out)
			test_running_loss.append(test_loss.data.cpu().numpy())

			test_pred = np.argmax(test_scores_acc.data.cpu().numpy(), axis=1)
			test_target = test_out.data.cpu().numpy()
			test_acc = np.sum(test_pred == test_target) / test_pred.shape[0]
			tp_count = 0
			fp_count = 0
			tn_count = 0
			fn_count = 0
			for i in range(len(test_pred)):
				if test_pred[i] == 1:
					if test_target[i] == 1:
						tp_count += 1
					else:
						fp_count += 1
				else:
					if test_target[i] == 0:
						tn_count += 1
					else:
						fn_count += 1
			test_running_pre.append(0 if tp_count <= 0 else tp_count/(tp_count+fp_count))
			test_running_rec.append(0 if tp_count <= 0 else tp_count/(tp_count+fn_count))
			test_running_acc.append(test_acc)

			print ("------------------------------")
			my_model.hidden = my_model.init_hidden(len(cp))

			with torch.no_grad():
				crop_scores_loss = my_model(crop_in, training=True)
				crop_scores_acc = my_model(crop_in)

				crop_loss = loss_function(crop_scores_loss, crop_out)
			crop_running_loss.append(crop_loss.data.cpu().numpy())

			crop_pred = np.argmax(crop_scores_acc.data.cpu().numpy(), axis=1)
			crop_target = crop_out.data.cpu().numpy()
			crop_acc = np.sum(crop_pred == crop_target) / crop_pred.shape[0]
			tp_count = 0
			fp_count = 0
			tn_count = 0
			fn_count = 0
			for i in range(len(crop_pred)):
				if crop_pred[i] == 1:
					if crop_target[i] == 1:
						tp_count += 1
					else:
						fp_count += 1
				else:
					if crop_target[i] == 0:
						tn_count += 1
					else:
						fn_count += 1
			crop_running_pre.append(0 if tp_count <= 0 else tp_count/(tp_count+fp_count))
			crop_running_rec.append(0 if tp_count <= 0 else tp_count/(tp_count+fn_count))
			crop_running_acc.append(crop_acc)

			if test_acc > best_acc[0]:
				saveModel(my_model)
				best_acc[0] = test_acc
				text_file = open("Best_Model_params.txt", "w")
				text_file.write("Ephocs: " + str(epoch + 1) + "\n")
				text_file.write("Hidden layer size: " + str(h_dim) + "\n")
				text_file.write("Drop out rate: " + str(do) + "\n")
				text_file.write("Learning rate: " + str(lr_in) + "\n")
				text_file.write("Weight Decay rate: " + str(wd_in) + "\n")
				text_file.write("Test accuracy: " + str(test_running_acc[-1]) + "\n")
				text_file.write("CropPAL accuracy: " + str(crop_running_acc[-1]) + "\n")
				text_file.write("CropPAL Precision: " + str(crop_running_pre[-1]) + "\n")
				text_file.write("CropPAL Recall: " + str(crop_running_rec[-1]) + "\n")
				text_file.write("Training accuracy: " + str(val_running_acc[-1]) + "\n")
				text_file.write("Precision: " + str(test_running_pre[-1]) + "\n")
				text_file.write("Recall: " + str(test_running_rec[-1]) + "\n")
				#text_file.write("F1: " + str((test_running_pre[-1]*test_running_rec[-1])/(test_running_pre[-1]+test_running_rec[-1])) + "\n")
				text_file.close()

	#my_model = loadModel(my_model)
	my_model.hidden = my_model.init_hidden(len(test_data))
	
	with torch.no_grad():
		test_scores = my_model(test_in)
		test_pred = np.argmax(test_scores.data.cpu().numpy(), axis=1)

	examples = list(torch.transpose(test_in.data.cpu(),0,1).numpy())
	preds = list(test_pred)
	targets = list(test_out.data.cpu().numpy())

	for i in range(len(examples)):
		sample = examples[i]
		examples[i] = (" ".join(IndexsToSentences(sample,vocab_rev)))

	output_samples = {"Inputs":examples, "Predictions":preds, "Targets":targets, "PMID":test_pmid}

	df = pd.DataFrame(data=output_samples,columns=["Inputs","Predictions","Targets","PMID"])
	df.to_csv("visual_confirm.csv",index=False)

	my_model.hidden = my_model.init_hidden(len(cp))
	
	with torch.no_grad():
		crop_scores = my_model(crop_in)
		crop_pred = np.argmax(crop_scores.data.cpu().numpy(), axis=1)

	examples = list(torch.transpose(crop_in.data.cpu(),0,1).numpy())
	preds = list(crop_pred)
	targets = list(crop_out.data.cpu().numpy())

	for i in range(len(examples)):
		sample = examples[i]
		examples[i] = (" ".join(IndexsToSentences(sample,vocab_rev)))

	output_samples = {"Inputs":examples, "Predictions":preds, "Targets":targets, "PMID":pmid_cp}

	df = pd.DataFrame(data=output_samples,columns=["Inputs","Predictions","Targets","PMID"])
	df.to_csv("visual_confirm_crop.csv",index=False)

	fig_name = "crop: lr=" + str(lr_in) + ",h_dim=" + str(h_dim) + ",do=" + str(do) + ",acc={:.4f}".format(test_running_acc[-1]) + ",lr_decay=" + str(lr_d) + ",split=" + str(split) + ",wd=" + str(wd_in) + ",pre={:.4f}".format(test_running_pre[-1]) + ",rc={:.4f}".format(test_running_rec[-1]) + ",ls={:.4f}".format(test_running_loss[-1])
	output_results = {"train_acc":train_running_acc, "train_loss":train_running_loss, "val_acc":val_running_acc, "val_loss":val_running_loss, "test_acc":test_running_acc, "test_precision":test_running_pre, "test_recall":test_running_rec, "crop_acc":crop_running_acc, "crop_precision":crop_running_pre, "crop_recall":crop_running_rec}
	results_df = pd.DataFrame(data=output_results,columns=["train_acc","train_loss","val_acc","val_loss","test_acc","test_precision","test_recall","crop_acc","crop_precision","crop_recall"])
	results_df.to_csv("./results2/"+fig_name+".csv",index=False)

	fig_1 = plt.figure(fig_name + "_1")
	plt.subplot(2,1,1)
	plt.ylim(0,1.5)
	plt.plot(train_running_loss, alpha = 0.5, label="Training Loss")
	plt.plot(val_running_loss, alpha = 0.5, label="Validation Loss")
	plt.ylabel("LOSS")
	plt.xlabel("BATCHES (5 batches per epoch)")
	plt.legend()
	plt.grid()

	plt.subplot(2,1,2)
	plt.plot(train_running_acc, alpha = 0.5, label="Training Accuracy")
	plt.plot(val_running_acc, alpha = 0.5, label="Validation Accuracy")
	plt.plot(test_running_acc, alpha = 0.5, label="Testing Accuracy")
	plt.plot(crop_running_acc, alpha = 0.5, label="CropPAL Accuracy")
	plt.ylabel("ACCURACY")
	plt.xlabel("BATCHES (5 batches per epoch)")
	plt.legend()
	plt.grid()

	fig_2 = plt.figure(fig_name + "_2")
	plt.subplot(2,1,1)
	plt.plot(train_running_pre, alpha = 0.5, label="Training Precision")
	plt.plot(val_running_pre, alpha = 0.5, label="Validation Precision")
	plt.plot(test_running_pre, alpha = 0.5, label="Testing Precision")
	plt.plot(crop_running_pre, alpha = 0.5, label="CropPAL Precision")
	plt.ylabel("PRECISION")
	plt.xlabel("BATCHES (5 batches per epoch)")
	plt.legend()
	plt.grid()

	plt.subplot(2,1,2)
	plt.plot(train_running_rec, alpha = 0.5, label="Training Recall")
	plt.plot(val_running_rec, alpha = 0.5, label="Validation Recall")
	plt.plot(test_running_rec, alpha = 0.5, label="Testing Recall")
	plt.plot(crop_running_rec, alpha = 0.5, label="CropPAL Recall")
	plt.ylabel("RECALL")
	plt.xlabel("BATCHES (5 batches per epoch)")
	plt.legend()
	plt.grid()

	plt.show()
	
	fig_1.savefig("./results2/" + fig_name + "_1.png")
	fig_2.savefig("./results2/" + fig_name + "_2.png")

	plt.clf()

	


#__________________________End of main____________________________
def saveModel(model):
	torch.save(model.state_dict(),"classifier.pt")

def loadModel(model):
	model.load_state_dict(torch.load("classifier.pt"))	
	return model

def lr_decay(optimizer, epoch, start_lr, lr_decay_time):

	lr = start_lr * (0.1**(epoch // lr_decay_time))

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	
	return optimizer

def test(data, vocab, my_model):

	test_in, test_out = paddInputs(data,vocab)
	test_in = autograd.Variable(torch.transpose(torch.LongTensor(test_in).cpu(),0,1))
	test_out = autograd.Variable(torch.LongTensor(test_out).cpu())

	my_model.hidden = my_model.init_hidden(len(data))

	test_scores = my_model(test_in)

	test_pred = np.argmax(test_scores.data.cpu().numpy(), axis=1)
	test_target = test_out.data.cpu().numpy()
	test_acc = np.sum(test_pred == test_target) / test_pred.shape[0]
	tp_count = 0
	fp_count = 0
	tn_count = 0
	fn_count = 0
	for i in range(len(test_pred)):
		if test_pred[i] == 1:
			if test_target[i] == 1:
				tp_count += 1
			else:
				fp_count += 1
		else:
			if test_target[i] == 0:
				tn_count += 1
			else:
				fn_count += 1
	precision = (0 if tp_count <= 0 else tp_count/(tp_count+fp_count))
	recall = (0 if tp_count <= 0 else tp_count/(tp_count+fn_count))
	accuracy = test_acc

	return (accuracy, precision, recall)


def sentenceToIndexs(sentence,vocab):

	sent_to_index = []

	for i in range(len(sentence)):		
		if sentence[i] in vocab:
			sent_to_index.append(vocab[sentence[i]])
		else:
			sent_to_index.append(vocab["_UNKNOWN_"])

	return sent_to_index

def IndexsToSentences(sentence,vocab):

	index_to_sent = []

	for i in range(len(sentence)):		
		if sentence[i] in vocab:
			index_to_sent.append(vocab[sentence[i]])
		else:
			index_to_sent.append("_UNKNOWN_")

	return index_to_sent

def paddInputs(inputs,vocab):
	in_data = []
	out_data = []

	max_len = 0

	for sample in inputs:
		if len(sample[0]) > max_len:
			max_len = len(sample[0])

	for sample in inputs:
		padding = [vocab["_PAD_"]]*(max_len-len(sample[0]))
		in_data.append(padding + sample[0])
		out_data.append(sample[1])

	return in_data, out_data

def getFold(fold):

	max_len = 0

	in_data = []
	out_data = []

	for sample in fold:
		in_data.append(sample[0])
		out_data.append(sample[1])

		text_len = len(sample[0])
		if text_len > max_len:
			max_len = text_len

	for i in range(len(in_data)):
		sample = in_data[i]
		padding = ["_PAD_"]*(max_len-len(sample))
		new = padding + sample
		in_data[i] = new


	return in_data, out_data



def getData():
	input_directory = "Data/"

	input_data_ret = []
	output_data_ret = []
	pmid_data_ret = []

	for filename in os.listdir(input_directory):
		if filename != "simple_subset_1.csv":
			continue
		file_path = os.path.join(input_directory, filename)
		# print (filename)
		input_data, output_data, pmid_data = parseDataFile(file_path)
		input_data_ret += input_data[:]
		output_data_ret += output_data[:]
		pmid_data_ret += pmid_data[:]

	for i in range(len(input_data_ret)):
		input_data_ret[i] = ["_SOS_"] + input_data_ret[i] + ["_EOS_"]

	for i in range(len(output_data_ret)):
		if output_data_ret[i] == "Y":
			output_data_ret[i] = 1
		else:
			output_data_ret[i] = 0

	return input_data_ret, output_data_ret , pmid_data_ret

def getTestData():
	input_directory = "Data/"

	input_data_ret = []
	output_data_ret = []
	pmid_data_ret = []

	for filename in os.listdir(input_directory):
		if filename != "simple_subset_cp.csv":
			continue
		file_path = os.path.join(input_directory, filename)
		print (filename)
		input_data, output_data, pmid_data = parseDataFile(file_path)
		input_data_ret += input_data[:]
		output_data_ret += output_data[:]
		pmid_data_ret += pmid_data[:]

	for i in range(len(input_data_ret)):
		input_data_ret[i] = ["_SOS_"] + input_data_ret[i] + ["_EOS_"]

	for i in range(len(output_data_ret)):
		if output_data_ret[i] == "Y":
			output_data_ret[i] = 1
		else:
			output_data_ret[i] = 0

	return input_data_ret, output_data_ret , pmid_data_ret


def parseDataFile(file_path):
	input_data = []
	output_data = []
	pmid_data = []
	
	df = pd.read_csv(file_path, index_col=False, header=0)

	input_data = df["Input"].tolist()
	output_data = df["Output"].tolist()
	pmid_data = [str(i) for i in df["PMID"].tolist()]

	for i in range(len(input_data)):
		input_data[i] = input_data[i].split()

	return input_data, output_data, pmid_data

def getWordVectors():
	df = pd.read_csv("word_vectors.csv", index_col=False, header=0, keep_default_na=False)

	words = df["Words"].tolist()
	vectors = df["Vectors"].tolist()

	vectors_ret = []
	for vector in vectors:
		temp_vector = vector.split()
		temp_vector_float = []
		for elem in temp_vector:
			temp_vector_float.append(float(elem))
		vectors_ret.append(temp_vector_float)

	words_ret = {}
	words_ret_rev = {}

	for i in range(len(words)):
		words_ret[str(words[i])] = int(i)

	for key in words_ret:
		words_ret_rev[words_ret[key]] = key


	return words_ret, words_ret_rev, vectors_ret

def chunks(l,k):

	shuffle(l)

	ret = []

	rem = len(l)/k
	n = 0

	if (rem - int(rem)) > 0:
		n = int(rem) + 1
	else:
		n = int(rem)

	for i in range(0,len(l), n):
		ret.append(l[i:i+n])

	return ret

def trainTestSplit(data,split):

	train_split = split

	neg_data = []
	pos_data = []

	neg_data_dic = {}
	pos_data_dic = {}

	neg_data_dic_sort = []
	pos_data_dic_sort = []

	neg_train_data = []
	pos_train_data = []
	neg_test_data = []
	pos_test_data = []

	for sample in data:
		if sample[1] == 1:
			pos_data.append(sample)
			if sample[2] in pos_data_dic:
				pos_data_dic[sample[2]] += [sample]
			else:
				pos_data_dic[sample[2]] = [sample]
		else:
			neg_data.append(sample)
			if sample[2] in neg_data_dic:
				neg_data_dic[sample[2]] += [sample]
			else:
				neg_data_dic[sample[2]] = [sample]

	neg_train_count = int(len(neg_data)*train_split)
	pos_train_count = int(len(pos_data)*train_split)
	
	for key in neg_data_dic:
		neg_data_dic_sort.append((len(neg_data_dic[key]), key))
	for key in pos_data_dic:
		pos_data_dic_sort.append((len(pos_data_dic[key]), key))

	neg_data_dic_sort = sorted(neg_data_dic_sort, key=lambda x:x[0])
	pos_data_dic_sort = sorted(pos_data_dic_sort, key=lambda x:x[0])


	counter = 0

	for sample in neg_data_dic_sort:
		counter += sample[0]
		if counter <= neg_train_count:
			neg_train_data += neg_data_dic[sample[1]]
		else:
			neg_test_data += neg_data_dic[sample[1]]

	counter = 0

	for sample in pos_data_dic_sort:
		counter += sample[0]
		if counter <= pos_train_count:
			pos_train_data += pos_data_dic[sample[1]]
		else:
			pos_test_data += pos_data_dic[sample[1]]

	test = neg_test_data + pos_test_data
	train = neg_train_data + pos_train_data

	test = [(x[0],x[1],x[2]) for x in test]
	train = [(x[0],x[1],x[2]) for x in train]

	return train, test

def loadPTModel(model):
	model.load_state_dict(torch.load("_ptmp_v1.pt"))	
	return model

def loadVocab():
	df = pd.read_csv("_ptmwi_v1.csv")

	vocab = {}
	for idx,row in df.iterrows():
		vocab[str(row.loc["Token"])] = int(row.loc["Index"])

	return vocab


if __name__ == "__main__":
	main()
