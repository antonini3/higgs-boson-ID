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




	print len(fpr)
	print len(tpr)

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

if __name__ == '__main__':
	read()

r = np.load(open('../results/crashing_stats_dumb', 'rb'))
plt.plot(range(r.shape[0]), r)
# plt.plot(np.load(), np.load(open('../roc_data/pull_tpr', 'rb')), label='FDA on pull data (area = %0.3f)' % auc(np.load(open('../roc_data/pull_fpr', 'rb')), np.load(open('../roc_data/pull_tpr', 'rb'))))

