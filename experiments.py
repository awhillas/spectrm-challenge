from __future__ import print_function

import os.path
from collections import OrderedDict

import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Convolution1D, MaxPooling1D, Flatten, Dropout, Merge
from keras.models import load_model

import models
from corpus import Formare, BlobFormare
from Results import Result


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Multi processing stuff

experiment = None  # Global that will hold the experiment instance for the following functions...

def trainer(data):
	""" Multiprocessing training function
	Coz the multiprocessing lib can only all functions defined at the modules top level
	"""
	return experiment.train(data)

def tester(data):
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
			print("Doing training")
			model.train(self.training_data())
			model.save()
		else:
			model.load()

		# Test the model.
		if self.do_testing:
			print("Doing testing")
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
		return self.data
		# return {  # dev, small set to iron out bugs first
		# 	'missing': OrderedDict(sorted(self.data['missing'].items())[:1000]),
		# 	'dialogs': OrderedDict(sorted(self.data['dialogs'].items())[:1000])
		# }


class BaseLineAvg(BaseLine):
	ModelClass = models.NoModelAvg


class BaseLineRareWords(BaseLine):
	ModelClass = models.NoModelRareWords


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class MLP(object):
	"""
	Simple Multi-Layer Perceptron (MPL) approach.
	"""
	def __init__(self, corpus):
		self.corpus = corpus
		self.nb_epoch = 2
		self.cv_split = 0.2

	def run(self):

		print("Formatting data...")
		((X_train, y_train), (X_test, y_test)) = self.get_data()

		print('Build model...')
		model = self.build_model()

		print('Train...')
		model.fit(numpy.array(X_train), y_train,
				  nb_epoch=self.nb_epoch,
				  validation_data=(X_test, y_test))

		print('Test...')
		score, acc = model.evaluate(X_test, y_test)
		print('\nTest score:', score)
		print('Test accuracy:', acc)

	def get_data(self):
		return self.corpus.sentence_triples_cbow()

	def build_model(self):
		model = Sequential([
			Dense(999, input_dim=300*3),
			Dense(500),
			Dense(50),
			Dense(1),
			Activation('sigmoid')
		])
		# try using different optimizers and different optimizer configs
		model.compile(loss='binary_crossentropy',
					  optimizer='adam',
					  metrics=['accuracy'])
		return model

class CNN(MLP):
	"""
	Get all the utterances from the dialog, concatenate into a blob and compare it with the missing sentence.
	1% on 1000 training/testing set.
	"""
	def __init__(self, corpus):
		super(CNN, self).__init__(corpus)
		self.max_features = len(corpus.nlp.vocab)
		self.maxlen_missing = 30
		self.maxlen_dialogs = self.maxlen_missing * 5
		self.miss_form = Formare(max_length=self.maxlen_missing)
		self.dialog_form = BlobFormare(max_length=self.maxlen_dialogs)
		self.embedding_dims = 50
		self.nb_filter = 250
		self.filter_length = 3
		self.hidden_layer_dim = 50  # was 250
		self.cv_split = 0.1
		self.nb_epoch = 3
		self.model_file = "CNN.model.h5"

	def run(self):
		print("Formatting data...")
		X_train_dialogs, X_train_missing, Y, test_corpus = self.get_data()

		if os.path.isfile(self.model_file):
			print('loading model from file '+self.model_file)
			model = load_model(self.model_file)
		else:
			print('Build model...')
			model = self.build_model()

			print('Train...')
			model.fit([X_train_dialogs, X_train_missing], Y,
					  nb_epoch=self.nb_epoch,
					  validation_split=0.1)

			model.save(self.model_file)

		print('Test...')
		return self.test(model, test_corpus)

	def test(self, model, corpus):
		"""
		Call the predict() method on the model over a part of the training data.
		For every
		:param model: Model to test. Must have a .predict() method.
		:param corpus: Corpus to run tests against
		:return: Result object.
		"""
		def max_key(d):
			v = list(d.values())
			k = list(d.keys())
			return k[v.index(max(v))]

		theMissing = corpus.fashion_missing(self.miss_form)
		theDialogs = corpus.fashion_dialogs(self.dialog_form)
		# for each missing utter. generate a prediction against each dialog and
		predictions = {}
		for m_key, m in theMissing.iteritems():
			ds = theDialogs.values()
			ms = [m] * len(ds)
			ds, ms = map(numpy.array, [ds, ms])
			predictions[m_key] = model.predict_proba([ds, ms])
		# Choose the one with the highest
		# print("predictions", predictions)
		lookup = theDialogs.keys()
		guesses = [lookup[p.flatten().argmax()] for answer, p in predictions.iteritems()]
		# print("Accuracy: {0:.2f}".format(guesses.count(True) / len(guesses) * 100))
		return Result(guesses, theMissing.keys())

	def get_data(self):
		train_corpus, test_corpus = self.corpus.split()

		X_train_dialogs = train_corpus.fashion_dialogs(self.dialog_form).values() * 2

		X_train_missing = train_corpus.get_ham(self.miss_form)  # positive examples
		X_train_missing += train_corpus.get_spam(self.miss_form) # negatives

		Y = numpy.array([1.0] * (len(X_train_missing) / 2) + [0.0] * (len(X_train_missing) / 2))

		return numpy.array(X_train_dialogs), numpy.array(X_train_missing), Y, test_corpus

	def build_model(self):

		# - - - - - - - - - - - - - - - - - Dialog Convolution

		dialog_model = Sequential()
		# we start off with an efficient embedding layer which maps
		# our vocab indices into embedding_dims dimensions
		dialog_model.add(Embedding(input_dim=self.max_features,  # vocab size
								   output_dim=self.embedding_dims,  # vector out size
								   input_length=self.maxlen_dialogs,
								   dropout=0.2))

		# we add a Convolution1D, which will learn nb_filter
		# word group filters of size filter_length:
		dialog_model.add(Convolution1D(nb_filter=self.nb_filter,
									   filter_length=self.filter_length,
									   border_mode='valid',
									   activation='relu',
									   subsample_length=1))
		# we use max pooling:
		dialog_model.add(MaxPooling1D(pool_length=dialog_model.output_shape[1]))

		# We flatten the output of the conv layer,
		# so that we can add a vanilla dense layer:
		dialog_model.add(Flatten())

		# - - - - - - - - - - - - - - - - - Missing utterance Convolution

		missing_model = Sequential()
		missing_model.add(Embedding(self.max_features,
									self.embedding_dims,
									input_length=self.maxlen_missing,
									dropout=0.2))
		missing_model.add(Convolution1D(nb_filter=self.nb_filter,
										filter_length=self.filter_length,
										border_mode='valid',
										activation='relu',
										subsample_length=1))
		missing_model.add(MaxPooling1D(pool_length=missing_model.output_shape[1]))
		missing_model.add(Flatten())

		# - - - - - - - - - - - - - - - - - Merge for MLP

		merged = Merge([dialog_model, missing_model], mode='concat')

		model = Sequential()
		model.add(merged)
		# We add a vanilla hidden layer:
		model.add(Dense(self.hidden_layer_dim))
		model.add(Dropout(0.2))
		model.add(Activation('relu'))

		# We project onto a single unit output layer, and squash it with a sigmoid:
		model.add(Dense(1))
		model.add(Activation('sigmoid'))

		model.compile(loss='binary_crossentropy',
					  optimizer='adam',
					  metrics=['accuracy'])
		return model


class CNN2(CNN):
	def __init__(self, corpus):
		super(CNN, self).__init__(corpus)
		self.model_file = "CNN2.model.h5"
		self.dialog_form = Formare(max_length=self.maxlen_missing)
		self.hidden_layer_dim = 250
		self.embedding_dims = 300

	def get_data(self):
		""" Get all the pairs of utterances and use them as training data. """
		train_corpus, test_corpus = self.corpus.split()

		X_train_dialogs = train_corpus.get_sandwiches(self.dialog_form) * 2

		X_train_missing = train_corpus.get_ham(self.miss_form)  # positive examples
		X_train_missing += train_corpus.get_spam(self.miss_form)  # negatives

		Y = numpy.array([1.0] * (len(X_train_missing) / 2) + [0.0] * (len(X_train_missing) / 2))

		return numpy.array(X_train_dialogs), numpy.array(X_train_missing), Y, test_corpus
