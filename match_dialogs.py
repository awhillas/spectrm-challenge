import sys
import re
from os import path
from collections import OrderedDict

import experiments
from experiments import BaseLine


# Data challenge stuff...
DATA_DIR = 'challenge_data'
TRAINING_DIALOGS_FILE = 'train_dialogs.txt'
TRAINING_MISSING_FILE = 'train_missing.txt'
TEST_OUTPUT_FILE = 'test_missing_with_predictions.txt'


def load_raw_data(filename, as_single_sentences = False):
	print "Reading raw data file ", filename
	with open(filename,'r') as f:
		return clean_raw_text(f.readlines(), as_single_sentences)

def clean_raw_text(raw_dialogs, one_sentence = False):
	"""
	Remove the cruff from the beginning of each line and organise text into dialogs
	indexed by ID. Parse with spaCy.
	"""
	print "Processing..."
	output = OrderedDict()
	for sentence in raw_dialogs:
		id, text = sentence.strip('\n').split(' +++$+++ ')
		s = re.sub('<[^<]+?>', '', text) # clean out html tags
		en = unicode(s, "utf-8")
		if not one_sentence:
			if id in output:
				output[id].append(en)
			else:
				output[id] = [en]
		else:
			output[id] = en
	return output

def save_results(results_list):
	with open(TEST_OUTPUT_FILE) as f:
		for item in f:
			f.write("%s\n" % item)


if __name__ == '__main__':
	"""
	This script should be called as
		python match_dialogs.py path/to/test_dialogs.txt path/to/test_missing.txt
	and write the predicted conversation numbers for all missing lines to a file
	named test_missing_with_predictions.txt
	"""
	# If True then a model is produced
	train = False
	test = False
	# Default experiment class to run. Should train on all training data.
	ExperimentClass = BaseLine

	# if called with file names, load data from there else load from default location / output an error
	if len(sys.argv) > 2:
		dialogs_file, missing_file = sys.argv[1], sys.argv[2]
	elif len(sys.argv) == 2:
		# Experimenting trying different modles.
		train = test = True
		ExperimentClass = getattr(experiments, sys.argv[1])  # so we can... experiment
		dialogs_file, missing_file = path.join(DATA_DIR, TRAINING_DIALOGS_FILE), path.join(DATA_DIR, TRAINING_MISSING_FILE)
	else:
		train = True
		dialogs_file, missing_file = path.join(DATA_DIR, TRAINING_DIALOGS_FILE), path.join(DATA_DIR, TRAINING_MISSING_FILE)
		print "please call this script with `python match_dialogs.py path/to/test_dialogs.txt path/to/test_missing.txt`"
		print "Loading default training data '{}' and '{}'".format(dialogs_file, missing_file)

	# Load the data

	# print "train", train, "test", test, sys.argv

	data = {'dialogs':{}, 'missing':{}}
	data['dialogs'] = load_raw_data(dialogs_file)
	data['missing'] = load_raw_data(missing_file, True)
	assert len(data['dialogs']) == len(data['missing'])
	print len(data['missing']), "Dialogs loaded"

	# Run the experiment(s)

	experiment = ExperimentClass(data, train, test)
	results = experiment.run()

	# We out!

	if test:
		print "Results:\n", results
	elif not train:
		save_results(results.guesses)
