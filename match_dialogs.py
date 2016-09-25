from __future__ import print_function

import sys
import re
from os import path
from collections import OrderedDict

import experiments
from corpus import Corpus
from experiments import BaseLine


# Data challenge stuff...
DATA_DIR = 'challenge_data'
TRAINING_DIALOGS_FILE = 'train_dialogs.txt'
TRAINING_MISSING_FILE = 'train_missing.txt'
TEST_OUTPUT_FILE = 'test_missing_with_predictions.txt'



if __name__ == '__main__':
	"""
	This script should be called as
		python match_dialogs.py path/to/test_dialogs.txt path/to/test_missing.txt
	and write the predicted conversation numbers for all missing lines to a file
	named test_missing_with_predictions.txt
	"""
	train = False  # If True then a model is produced
	clip_data_at = 0
	# Default experiment class to run. Should train on all training data.
	ExperimentClass = BaseLine

	# if called with file names, load data from there else load from default location / output an error
	if len(sys.argv) > 2:
		dialogs_file, missing_file = sys.argv[1], sys.argv[2]
		if len(sys.argv) > 3:
			# Experimenting...
			ExperimentClass = getattr(experiments, sys.argv[3])
			train = True
			if len(sys.argv) > 4:
				# Only use some of the data
				clip_data_at = int(sys.argv[4])
	else:
		train = True
		dialogs_file, missing_file = path.join(DATA_DIR, TRAINING_DIALOGS_FILE), path.join(DATA_DIR, TRAINING_MISSING_FILE)
		print("please call this script with `python match_dialogs.py path/to/test_dialogs.txt path/to/test_missing.txt`")
		print("Loading default training data '{}' and '{}'".format(dialogs_file, missing_file))

	# Load the data

	# print "TRAIN", train, "TEST", test, sys.argv  # debug
	data = Corpus(dialogs_file, missing_file, training=train, clip_data_at=clip_data_at)
	print(data.length, "Dialogs loaded")

	# Run the experiment(s)

	experiment = ExperimentClass(data)
	results = experiment.run()
	print(results)