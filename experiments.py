import os.path
from itertools import islice
from collections import OrderedDict
from multiprocessing import Pool

import spacy
# from spacy.parts_of_speech import NOUN, VERB, PROPN, ADJ
from numpy import dot
from numpy.linalg import norm

import models
from Results import Result

print "Loading English module..."
nlp = spacy.load('en')
# nlp = None


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Multi processing stuff

experiment = None  # Global that will hold the experiment instance for the following functions...

def trainer():
	""" Multiprocessing training function
	Coz the multiprocessing lib can only all functions defined at the modules top level
	"""
	return experiment.train()

def tester():
	""" Multiprocessing testing function """
	experiment.test()

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

	def __init__(self, data, train = True, test = False):
		self.data = data
		self.do_training = train  # train a model
		self.do_testing = test  # reserve 10% data for testing, don't save model.
		experiment = self

	@property
	def data_split_point(self):
		return int(len(self.data['missing']) * (1 - self.test_set_size))

	def training_data(self):
		return {
			'missing': dict(self.data['missing'].items()[:self.data_split_point]),
			'dialogs': dict(self.data['dialogs'].items()[:self.data_split_point])
		}

	def test_data(self):
		return {
			'missing': dict(self.data['missing'].items()[self.data_split_point:]),
			'dialogs': dict(self.data['dialogs'].items()[self.data_split_point:])
		}

	def run(self):
		"""
		Run Experiment: train, test, display [, save]
		The cycle all experiments go through.
		"""
		# train the model or use the one given.
		if self.do_training:
			model = self.train()

		# Test the model.
		if self.do_testing:
			return self.test(model, self.test_data())

	def train(self, data):
		""" Train the model """

		print("Training model...")
		model = ModelClass(nlp)
		model.train(data)
		# else:
		# 	model = self.load()

		# if not self.do_testing:  # we only train on 90% when testing so don't keep model
		# 	self.save()

		return model

	def test(self, model, data):
		guesses = []
		solutions = []
		i = 0
		for id, sentence in data['missing'].iteritems():
			i += 1
			print "{0:.1f}% {1}: {2}".format(float(i)/len(data['missing'])*100, id, sentence)
			prediction = model.predict(sentence, data['dialogs'], id)
			guesses.append(prediction)
			solutions.append(id)
		return Result(guesses, solutions)


class BaseLine(Experiment):
	"""
	Get a stupid baseline so we can see how much the learning is actually
	improving
	"""

	# This model isn't really a model.
	ModelClass = models.NoModelCBOW

	def train(self):
		print "Training model (sort of)"
		model = self.ModelClass(nlp)
		model.train(self.training_data())
		return model

	def training_data(self):
		return self.test_data()

	def test_data(self):
		""" NoModel requires no (real) training so use all data for testing """
		# return self.data
		return {
			'missing': dict(sorted(self.data['missing'].items())[:100]),
			'dialogs': dict(sorted(self.data['dialogs'].items())[:100])
		}

class BaseLineAvg(BaseLine):
	ModelClass = models.NoModelAvg

class BaseLineRareWords(BaseLine):
	ModelClass = models.NoModelRareWords
