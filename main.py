import argparse
import os
import sys

#TODO import more stuff
from models import disease,symptom

# Entry Script for Lung Sounds

#TODO modify test() and train() function signatures
def main(args):
	task = args.task
	if task != 'symptom' and task != 'disease':
		raise Exception('Task needs to be symptom or disease.')

	train = args.train
	test = args.test
	if not (train and test) or (train and test):
		raise Exception('Only one of train or test must be true.')

	if test:
		if not os.path.exists(args.model):
			raise Exception('Path to model is invalid.')
		if task == 'disease':
			disease.test(args.model)
		else:
			symptom.test(args.model)

	if train:
		if task == 'disease':
			disease.train()
		else:
			symptom.train()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', help='symptom or disease')
	parser.add_argument('--train', dest='train', action='store_true', help='train the model')
	parser.set_defaults(train=True)
	parser.add_argument('--test', dest='test', action='store_true', help='test the model')
	parser.set_defaults(test=False)
	parser.add_argument('--model', help='path to model, required if testing', required='--test' in sys.argv)
	main(parser.parse_args())

