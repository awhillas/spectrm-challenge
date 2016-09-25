from __future__ import print_function

import sys
from os import path

import experiments


# Data challenge stuff...
DATA_DIR = 'challenge_data'
TRAINING_DIALOGS_FILE = path.join(DATA_DIR, 'train_dialogs.txt')
TRAINING_MISSING_FILE = path.join(DATA_DIR, 'train_missing.txt')
TEST_OUTPUT_FILE = 'test_missing_with_predictions.txt'



if __name__ == '__main__':
	"""
	This script should be called as
		python match_dialogs.py path/to/test_dialogs.txt path/to/test_missing.txt
	and write the predicted conversation numbers for all missing lines to a file
	named test_missing_with_predictions.txt
	"""
	train = False  # If True then a model is produced and saved else its loaded from a file
	is_experiment = False
	clip_data_at = 0
	# Default experiment class to run. Should train on all training data.
	ExperimentClass = experiments.RareWords

	# if called with file names, load data from there else load from default location / output an error
	if len(sys.argv) > 2:
		dialogs_file, missing_file = sys.argv[1], sys.argv[2]
		if len(sys.argv) > 3:
			# Experimenting...
			train = True
			ExperimentClass = getattr(experiments, sys.argv[3])
			if len(sys.argv) > 4:
				is_experiment = True
				# Only use some of the data
				clip_data_at = int(sys.argv[4])
	else:
		dialogs_file, missing_file = TRAINING_DIALOGS_FILE, TRAINING_MISSING_FILE  # assume testing already trained model
		print("please call this script with `python match_dialogs.py path/to/test_dialogs.txt path/to/test_missing.txt`")
		print("Loading default training data '{}' and '{}'".format(dialogs_file, missing_file))

	# Run the experiment(s)

	experiment = ExperimentClass(dialogs_file, missing_file, train, is_experiment, clip_data_at)
	results = experiment.run()

	print(results)
	if not is_experiment:
		results.save_guesses(TEST_OUTPUT_FILE)