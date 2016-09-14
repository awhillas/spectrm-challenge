import os.path
from itertools import islice
from collections import OrderedDict
from multiprocessing import Pool

from numpy import dot
from numpy.linalg import norm

import models
from Results import Result



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Multi processing stuff

experiment = None  # Global that will hold the experiment instance for the following functions...

def trainer(data):
	""" Multiprocessing training function
	Coz the multiprocessing lib can only all functions defined at the modules top level
	"""
	return experiment.train(data)

def tester():
	""" Multiprocessing testing function """
	experiment.test(data)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Experiment(object):
	"""
	Interface that all experiments should impliment.
	Should handle training / testing data split,
	and loading / saving the model.
	"""
	# Override for save file names
	ModelClass = None
	test_set_size = 0.10  # % of data for testing.

	def __init__(self, data, train = True, test = True):
		self.data = data
		self.do_training = train
		self.do_testing = test
		experiment = self

	@property
	def data_split_point(self):
		return int(len(self.data['missing']) * (1 - self.test_set_size))

	def training_data(self):
		return {
			'missing': OrderedDict(self.data['missing'].items()[:self.data_split_point]),
			'dialogs': OrderedDict(self.data['dialogs'].items()[:self.data_split_point])
		}

	def test_data(self):
		return {
			'missing': OrderedDict(self.data['missing'].items()[self.data_split_point:]),
			'dialogs': OrderedDict(self.data['dialogs'].items()[self.data_split_point:])
		}

	def run(self):
		"""
		Run Experiment: train, test, display [, save]
		The cycle all experiments go through.
		"""
		# train the model or use the one given.
		model = self.ModelClass()
		if self.do_training:
			print "Doing training"
			model.train(self.training_data())
			model.save()
		else:
			model.load()

		# Test the model.
		if self.do_testing:
			print "Doing testing"
			guesses = model.test(self.test_data())
			return Result(guesses, self.test_data()['missing'].keys())

class BaseLine(Experiment):
	"""
	Get a stupid baseline so we can see how much the learning is actually
	improving
	"""

	# This model isn't really a model.
	ModelClass = models.NoModelCBOW

	# def train(self):
	# 	print "Training {} model.".format(type(BaseLine.ModelClass))
	# 	model = BaseLine.ModelClass()
	# 	model.train(self.training_data()['dialogs'].items())
	# 	return model

	def training_data(self):
		return self.test_data()

	def test_data(self):
		""" NoModel requires no (real) training so use all data for testing """
		# return self.data
		return {  # dev, small set to iron out bugs first
			'missing': OrderedDict(sorted(self.data['missing'].items())[:1000]),
			'dialogs': OrderedDict(sorted(self.data['dialogs'].items())[:1000])
		}


class BaseLineAvg(BaseLine):
	ModelClass = models.NoModelAvg


class BaseLineRareWords(BaseLine):
	ModelClass = models.NoModelRareWords
