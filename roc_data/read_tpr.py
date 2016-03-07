
def read():
	laNet = {}
	fpr = []
	tpr = []
	tpr_bool = False
	with open('tpr_fpr_laNet', 'r') as f:
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
	
	print len(fpr)
	print len(tpr)
	
	laNet_Two = {}
	fpr = []
	tpr = []
	tpr_bool = False
	with open('tpr_fpr_laNetTwo', 'r') as f:
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

	print len(fpr)
	print len(tpr)

if __name__ == '__main__':
	read()