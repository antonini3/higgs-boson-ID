from util import *

import numpy as np
import matplotlib.pyplot as plt

from mpltools import style
from mpltools import layout

style.use('ggplot')

def read():
	laNet = {}
	fpr = []
	tpr = []
	tpr_bool = False
	with open('../roc_data/tpr_fpr_laNet.lzh', 'r') as f:
		for line in f:
			numbers = line.split()
			if len(numbers) == 0:
				tpr_bool = True
			else:
				numbers = [float(n.strip()) for n in numbers]
				if tpr_bool:
					tpr += numbers
				else:
					fpr += numbers
	laNet['fpr'] = fpr
	laNet['tpr'] = tpr
	

	laNet_Two = {}
	fpr = []
	tpr = []
	tpr_bool = False
	with open('../roc_data/tpr_fpr_laNetTwo.lzh', 'r') as f:
		for line in f:
			numbers = line.split()
			if len(numbers) == 0:
				tpr_bool = True
			else:
				numbers = [float(n.strip()) for n in numbers]
				if tpr_bool:
					tpr += numbers
				else:
					fpr += numbers
	laNet_Two['fpr'] = fpr
	laNet_Two['tpr'] = tpr

	simple = {}
	fpr = []
	tpr = []
	tpr_bool = False
	with open('../roc_data/tpr_fpr_simple.lzh', 'r') as f:
		for line in f:
			numbers = line.split()
			if len(numbers) == 0:
				tpr_bool = True
			else:
				numbers = [float(n.strip()) for n in numbers]
				if tpr_bool:
					tpr += numbers
				else:
					fpr += numbers
	simple['fpr'] = fpr
	simple['tpr'] = tpr

	large_ass = {}
	fpr = []
	tpr = []
	tpr_bool = False
	with open('../roc_data/tpr_fpr_large_ass_network.lzh', 'r') as f:
		for line in f:
			numbers = line.split()
			if len(numbers) == 0:
				tpr_bool = True
			else:
				numbers = [float(n.strip()) for n in numbers]
				if tpr_bool:
					tpr += numbers
				else:
					fpr += numbers
	large_ass['fpr'] = fpr
	large_ass['tpr'] = tpr




	print len(fpr)
	print len(tpr)

	plt.plot(large_ass['fpr'], large_ass['tpr'], label='LaNetThree (area = %0.3f)' % auc(large_ass['fpr'], large_ass['tpr']))
	plt.plot(laNet_Two['fpr'], laNet_Two['tpr'], label='LaNetTwo (area = %0.3f)' % auc(laNet_Two['fpr'], laNet_Two['tpr']))
	plt.plot(laNet['fpr'], laNet['tpr'], label='LaNet (area = %0.3f)' % auc(laNet['fpr'], laNet['tpr']))
	plt.plot(simple['fpr'], simple['tpr'], label='SimpleCNN (area = %0.3f)' % auc(simple['fpr'], simple['tpr']))

	plt.plot(np.load(open('../roc_data/adaboost_fpr', 'rb')), np.load(open('../roc_data/adaboost_tpr', 'rb')), label='AdaBoost (area = %0.3f)' % auc(np.load(open('../roc_data/adaboost_fpr', 'rb')), np.load(open('../roc_data/adaboost_tpr', 'rb'))))
	plt.plot(np.load(open('../roc_data/pull_fpr', 'rb')), np.load(open('../roc_data/pull_tpr', 'rb')), label='FDA on pull data (area = %0.3f)' % auc(np.load(open('../roc_data/pull_fpr', 'rb')), np.load(open('../roc_data/pull_tpr', 'rb'))))


	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Higgs Boson Receiver Operating Characteristics')
	plt.legend(loc="lower right")
	plt.show()

def plot_peter():
	no_drop = list(reversed([0.805086,0.817657,0.816057,0.807371,0.816,0.813943,0.8172, 0.817829,0.806571,0.8196,0.803029,0.807314,
				0.800171, 0.801771, 0.807486, 0.803714, 0.809886, 0.808743,0.7988, 0.808286,0.781657,0.786057,0.7724, 0.785829, 0.782743, 0.771257, 0.50]))
	drop = list(reversed([0.815029,0.792457,0.803029, 0.816686, 0.811486,0.802971, 0.814971,0.812229,0.8116\
			,0.810229\
			,0.804171\
			, 0.812114\
			,0.809829\
			,0.804343\
			,0.801886\
			,0.8072\
			, 0.792171\
			, 0.798914\
			, 0.793886\
			, 0.798343\
			, 0.793657\
			,0.7904\
			, 0.784114\
			, 0.766914\
			, 0.501429\
			, 0.501429, 0.50]))

	epochs = range(0, len(drop))
	# print len(epochs)
	# print len(drop)
	# print len(no)
	plt.plot(epochs, no_drop, label='No Dropout')
	plt.plot(epochs, drop, label='Dropout (p = 0.5)')
	plt.xlim([0, 26])
	# plt.ylim([0.0, 1.0])
	plt.xlabel('Epochs')
	plt.ylabel('Validation accuracy')
	plt.title('Dropout Validation Set Accuracy')
	plt.legend(loc="lower right")
	plt.show()

def plot_no_overfitting():
	train_acc = [0.515625, 0.546875, 0.53125, 0.51875000000000004, 0.51249999999999996, 0.5, 0.50312500000000004, 0.47187499999999999, 0.49062499999999998, 0.48125000000000001, 0.52500000000000002, 0.546875, 0.74062499999999998, 0.78125, 0.75312500000000004, 0.83750000000000002, 0.80312499999999998, 0.80000000000000004, 0.78437500000000004]
	val_acc = [0.501714289188, 0.498285710812, 0.501714289188, 0.501714289188, 0.501714289188, 0.498285710812, 0.501714289188, 0.498285710812, 0.498285710812, 0.501714289188, 0.501714289188, 0.501714289188, 0.688571453094, 0.772400021553, 0.751028597355, 0.79240000248, 0.791885733604, 0.785428583622, 0.799428582191]
	epochs = range(1, len(val_acc) + 1)
	print len(train_acc)
	print len(epochs)
	print len(val_acc)
	# print len(epochs)
	# print len(drop)
	# print len(no)
	plt.plot(epochs, train_acc, label='Training accuracy')
	plt.plot(epochs, val_acc, label='Validation accuracy')
	plt.xlim([0, 20])
	# plt.ylim([0.0, 1.0])
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('Accuracies with 0.5 Dropout')
	plt.legend(loc="lower right")
	plt.show()
if __name__ == '__main__':
	# read()
	# plot_peter()
	plot_no_overfitting()


# plt.plot(np.load(), np.load(open('../roc_data/pull_tpr', 'rb')), label='FDA on pull data (area = %0.3f)' % auc(np.load(open('../roc_data/pull_fpr', 'rb')), np.load(open('../roc_data/pull_tpr', 'rb'))))

