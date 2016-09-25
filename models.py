from __future__ import print_function  # py3 print in py2
import operator
import random
try:
   import cPickle as pickle
except:
   import pickle
from collections import defaultdict
from multiprocessing import Pool
from collections import OrderedDict


import numpy
from numpy.linalg import norm
import spacy
from spacy.parts_of_speech import IDS as POS_tags

# print("Loading English module...")
# nlp = spacy.load('en')

# cosine to measue the similarity of two vectors
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
		# Always get a fresh worker pool initialized with the latest.
		return Pool(initializer = init, initargs = (self, ))

	@classmethod
	def train(cls, data):
		""" Train the model on the given data """
		pass

	def predict(self, sentence, dialogs):
		""" Given a sentence find the ID of the dialog that it matches """
		pass

	@classmethod
	def save(cls):
		print("Saving model to ", cls.save_filename)
		with open(cls.save_filename, 'w') as f:
			pickle.dump(self)

	@classmethod
	def load(cls):
		print("Loading model from ", cls.save_filename)
		with open(cls.save_filename, 'r') as f:
			return pickle.load(f)


def aggregate_vectors((id, dialog)):
	# dialog == (dialog_id, [u'sentence', u'sentence', ...])
	sentences = []
	pos_vector_dict = {}
	for s in dialog:
		try:
			sentences = nlp(s, parse=False)
			pos_vector_dict = model.group_by_pos(sentences)
		except:
			print("Problem tagging sentence:", s)

	return (id, pos_vector_dict)

def predictor((id, sentence)):
	return model.predict(sentence)

class NoModelCBOW(Model):
	"""
	Mesaure the cosine distance of word embeddings in each POS tag group.
	Simply sum them together to get a distance metric (unweighted).
	"""

	save_filename = "NoModelCBOW.pickle"

	def __init__(self):
		self.dialog_pos_vecs_cache = {}

	def train(self, data):
		self.dialog_pos_vecs_cache = OrderedDict(self.pool.map(aggregate_vectors, data['dialogs'].iteritems()))

	def test(self, data):
		return self.pool.map(predictor, data['missing'].iteritems())

	def test_alt(self, data):
		guesses = []
		solutions = []
		i = 0
		print ("Testing...")
		self.pool.map
		for id, sentence in data['missing'].iteritems():
			i += 1
			print ("{0:.1f}% {1}: {2}".format(float(i)/len(data['missing'])*100, id, sentence))
			prediction = self.predict(sentence, data['dialogs'])
			guesses.append(prediction)
			solutions.append(id)
		return guesses

	def predict(self, sentence):
		try:
			missing = nlp(sentence, parse=False)
		except:
			print ("Problem tagging sentence: {}".format(sentence))
			return random.choice(self.dialog_pos_vecs_cache.keys())

		predictions = {}
		for id, dialog_vectors in self.dialog_pos_vecs_cache.iteritems():
			predictions[id] = self.distance(missing, dialog_vectors)

		guess = max(predictions.iteritems(), key=operator.itemgetter(1))[0]
		# print ("Guess: {0}, distance: {1:.4f}".format(guess, float(predictions[guess])))
 		return guess
		# return max(predictions.iteritems(), key=operator.itemgetter(1))[0]

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

	def group_by_pos(self, sentence):
		""" CBOW word embeddings grouped by POS tags """
		pos_vector_dict = {}
		for word in sentence:
			if numpy.count_nonzero(word.vector) > 0:  # filter all zero vectors
				if word.pos_ in pos_vector_dict:
					pos_vector_dict[word.pos_] = pos_vector_dict[word.pos_] + word.vector
				else:
					pos_vector_dict[word.pos_] = word.vector
		return pos_vector_dict

	def save(self):
		print("Saving model to ", self.save_filename)
		with open(self.save_filename, 'w') as f:
			pickle.dump(self, f)

	def load(self):
		print("Loading model from ", self.save_filename)
		with open(self.save_filename, 'w') as f:
			return pickle.load(f)  # Class data...


class NoModelRareWords(NoModelCBOW):
	def group_by_pos(self, sentence):
		"""
		CBOW word embeddings
		grouped by POS tags
		filtering out words with a log frequency greater than -8.0
		"""
		pos_vector_dict = {}
		for word in sentence:
			if word.prob < -8.0 and numpy.count_nonzero(word.vector) > 0:  # filter all zero vectors
				if word.pos_ in pos_vector_dict:
					pos_vector_dict[word.pos_] = pos_vector_dict[word.pos_] + word.vector
				else:
					pos_vector_dict[word.pos_] = word.vector
		return pos_vector_dict

class NoModelAvg(NoModelCBOW):
	def group_by_pos(self, sentence):
		"""
		Averaged word embeddings
		grouped by POS tags
		filtering out words with a log frequency greater than -8.0
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
