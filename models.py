from __future__ import print_function  # py3 print in py2
import operator
import random
import cPickle as pickle
from multiprocessing import Pool
from collections import OrderedDict

import numpy
from numpy.linalg import norm
from spacy.parts_of_speech import IDS as POS_tags

from corpus import nlp, Dialog
from ga import GeneticAlgorithm, GASolution

# cosine to measure the similarity of two vectors
cosine = lambda v1, v2: numpy.dot(v1, v2) / (norm(v1) * norm(v2))

model = None  # used by the multiprocessing functions
def init(arg):
	""" initializer for sub-processes. """
	global model
	model = arg

class Model(object):
	"""
	Interface for all predictive models
	"""
	save_filename = "Model.pickle"

	@property
	def pool(self):
		# Always get a fresh worker pool initialized with the latest model.
		return Pool(initializer = init, initargs = (self, ))

	@classmethod
	def train(cls, data):
		""" Train the model on the given data """
		pass

	def predict(self, utterance):
		""" Given a sentence find the ID of the dialog that it matches """
		pass

	@classmethod
	def save(cls):
		print("Saving model to ", cls.save_filename)
		with open(cls.save_filename, 'w') as f:
			pickle.dump(model)

	@classmethod
	def load(cls):
		print("Loading model from ", cls.save_filename)
		with open(cls.save_filename, 'r') as f:
			return pickle.load(f)


def aggregate_vectors((id, dialog)):
	""" Aggregates word vectors by POS tag for a give dialog.
	dialog == (dialog_id, [u'sentence', u'sentence', ...])
	:return: (dialog_id, dict())
	"""
	return (id, dialog.group_dialogs_by_pos(filter_prob=model.filter_prob))

def predictor(sentence):
	return model.predict(sentence)


class ClusterUnsupervised(Model):
	"""
	Mesaure the cosine distance of word embeddings in each POS tag group.
	Simply sum them together to get a distance metric (unweighted).
	"""

	save_filename = "ClusterUnsupervised.pickle"
	dialog_pos_vecs_cache = None

	def __init__(self):
		super(ClusterUnsupervised, self).__init__()  # just in case
		self.filter_prob = 0.0

	def train(self, corpus):
		""" Not really training. Building a cache of the dialogs grouped by POS tag as vectors """
		if ClusterUnsupervised.dialog_pos_vecs_cache is None:
			if len(corpus) > 50:
				ClusterUnsupervised.dialog_pos_vecs_cache = OrderedDict(self.pool.map(aggregate_vectors, corpus.dialogs.iteritems()))
			else:  # i.e. debugging
				ClusterUnsupervised.dialog_pos_vecs_cache = OrderedDict(map(lambda (i, x): (i, x.group_dialogs_by_pos(self.filter_prob)), corpus.dialogs.iteritems()))

	def test(self, raw_utterances):
		""" Call predict for all the utterances passed. """
		if len(raw_utterances) > 50:
			return self.pool.map(predictor, raw_utterances)
		else:  # i.e. debugging
			return map(self.predict, raw_utterances)

	def predict(self, utterance):
		try:
			missing = nlp(utterance, parse=False)
		except:
			print ("Problem tagging sentence: {}".format(utterance))
			return random.choice(ClusterUnsupervised.dialog_pos_vecs_cache.keys())

		predictions = {}
		for i, dialog_vectors in ClusterUnsupervised.dialog_pos_vecs_cache.iteritems():
			predictions[i] = self.distance(missing, dialog_vectors)

		guess = max(predictions.iteritems(), key=operator.itemgetter(1))[0]
		return guess

	def distance(self, missing, dialog_pos_vecs):
		"""
		Distance measure between a missing utterence and a given dialog.
		"""
		pos_distances = dict((pos, 0.0) for pos in POS_tags.keys())

		missing_pos_vecs = self.group_by_pos(missing)

		for pos, _ in pos_distances.iteritems():
			if pos in missing_pos_vecs and pos in dialog_pos_vecs:
				# take the cosine distance between vectors of words with same POS
				pos_distances[pos] = cosine(missing_pos_vecs[pos], dialog_pos_vecs[pos])
			elif pos in missing_pos_vecs or pos in dialog_pos_vecs:
				# POS is in one of the missing sentence or the dialog but not both
				pos_distances[pos] = 0.0 # -1.0  # divergent
			else:
				# POS is in nether
				pos_distances[pos] = 0.0  # No information.
		# print("S: {}\nDs: {}".format(missing, pos_distances))
		return sum(pos_distances.values())

	def group_by_pos(self, doc):
		""" CBOW word embeddings grouped by POS tags """
		return Dialog.group_by_pos(doc, filter_prob=self.filter_prob)

	def save(self):
		print("Saving model to ", self.save_filename)
		with open(self.save_filename, 'w') as f:
			pickle.dump(self, f)

	def load(self):
		print("Loading model from ", self.save_filename)
		with open(self.save_filename, 'w') as f:
			return pickle.load(f)  # Class data...


class RareWords(ClusterUnsupervised):
	def __init__(self):
		super(RareWords, self).__init__()
		self.filter_prob = -8.0


class AverageVectors(ClusterUnsupervised):
	def group_by_pos(self, sentence):
		"""
		Averaged word embeddings
		grouped by POS tags
		filtering out words with a log frequency greater than -8.0
		turns out to be a bad idea :(
		"""
		pos_vecs = {}
		pos_vector_dict = {}
		# collect a list of vectors for each POS tag
		for word in sentence:
			if word.prob < -8.0 and numpy.count_nonzero(word.vector) > 0:  # filter all zero vectors
				if word.pos_ in pos_vecs:
					pos_vecs[word.pos_] += [word.vector]
				else:
					pos_vecs[word.pos_] = [word.vector]

		# Average the vectors for each POS tag
		for pos, vectors in pos_vecs.iteritems():
			if len(vectors) > 0:
				pos_vector_dict[pos] = sum(vectors) / len(vectors)
		return pos_vector_dict


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - Genetic Algorithm


class ClusterUnsupervisedGASolution(GASolution, RareWords):
	corpus = None

	def __init__(self, gen=None):
		super(ClusterUnsupervisedGASolution, self).__init__(gen)
		self.SolutionClass = ClusterUnsupervisedGASolution
		self.train(ClusterUnsupervisedGASolution.corpus)

	def fitness(self):
		self.test(self.test_data())

	@staticmethod
	def factory(gen=None, corpus=None):
		if not corpus is None:
			ClusterUnsupervisedGASolution.corpus = corpus
		return ClusterUnsupervisedGASolution(gen)