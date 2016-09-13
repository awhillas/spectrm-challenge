import operator
from collections import defaultdict
import numpy
from numpy.linalg import norm

from spacy.parts_of_speech import IDS as POS_tags

# cosine to measue the similarity of two vectors
cosine = lambda v1, v2: numpy.dot(v1, v2) / (norm(v1) * norm(v2))
# def cosine(v1, v2):
# 	out = numpy.dot(v1, v2) / (norm(v1) * norm(v2))
# 	if out != out:  # nan test
# 		print v1, v2
# 		return 0
# 	else:
# 		return out

class Model(object):
	"""
	Interface for all predictive models
	"""
	save_filename = "Model.pickle"
	nlp = None

	def __init__(self, nlp):
		""" Save the spaCy nlp langugae model as it is BIG! """
		Model.nlp = nlp

	@classmethod
	def train(cls, data):
		""" Train the model on the given data """
		pass

	def predict(self, sentence, dialog):
		""" Given a sentence find the ID of the dialog that it matches """
		pass

	def save(self):
		print("Saving model to ", self.save_filename)
		with open(model.save_filename, 'w') as f:
			pickle.dump(self)

	def load(self):
		print("Loading model from ", self.save_filename)
		with open(self.save_filename, 'w') as f:
			return pickle.load(f)

	@staticmethod
	def random_key(d):
		import random
		return random.choice(d['dialogs'].keys())


class NoModelCBOW(Model):
	"""
	Mesaure the cosine distance of word embeddings in each POS tag group.
	Simply sum them together to get a distance metric (unweighted).
	"""

	def __init__(self, nlp):
		super(NoModelCBOW, self).__init__(nlp)
		self.dialog_pos_vecs_cache = {}  # cache the dialog vectors

	def train(self, data):
		# Build a cache of word vectors by POS tag for each dialog
		i = 0
		for id, dialog in data['dialogs'].iteritems():
			i += 1
			print "Progress: {0:.2f}%; {1} {2}: \r".format(float(i)/len(data['dialogs'])*100, i, id),
			if not id in self.dialog_pos_vecs_cache:
				self.dialog_pos_vecs_cache[id] = {}
				for utterence in dialog:
					try:
						s = Model.nlp(utterence)
						self._group_by_pos(s, self.dialog_pos_vecs_cache[id])
					except:
						pass

	def predict(self, sentence, dialogs, correct = None):
		try:
			missing = Model.nlp(sentence)
		except:
			return self.random_key(dialogs)

		predictions = {}
		for id, dialog in dict(dialogs).iteritems():
			predictions[id] = self._distance(missing, id, dialog)
			print predictions[id], "\r",
		guess = max(predictions.iteritems(), key=operator.itemgetter(1))[0]
		print "Guess: {0}, distance: {1:.4f}".format(guess, float(predictions[guess]))
		return guess

	def _distance(self, missing, id, dialog):
		"""
		Distance measure
		"""
		pos_distances = dict((id, -1.0) for id in POS_tags.keys()) # -1 if POS not occuring

		missing_pos_vecs = {}
		self._group_by_pos(missing, missing_pos_vecs)

		dialog_pos_vecs = self.dialog_pos_vecs_cache[id]

		for pos, _ in pos_distances.iteritems():
			if pos in missing_pos_vecs and pos in dialog_pos_vecs:
				# take the cosine distance between vectors of words with same POS
				pos_distances[pos] = cosine(missing_pos_vecs[pos], dialog_pos_vecs[pos])
			elif pos in missing_pos_vecs or pos in dialog_pos_vecs:
				# POS is in ether the missing sentence or the dialog but not both
				pos_distances[pos] = 0.0  # -1.0?
			else:
				# POS is in nether
				pos_distances[pos] = 0.0

		return sum(pos_distances.values())

	def _group_by_pos(self, sentence, pos_vector_dict):
		""" CBOW word embeddings grouped by POS tags """
		for word in sentence:
			if numpy.count_nonzero(word.vector) > 0:  # filter all zero vectors
				if word.pos_ in pos_vector_dict:
					pos_vector_dict[word.pos_] = pos_vector_dict[word.pos_] + word.vector
				else:
					pos_vector_dict[word.pos_] = word.vector


class NoModelRareWords(NoModelCBOW):
	def _group_by_pos(self, sentence, pos_vector_dict):
		"""
		CBOW word embeddings
		grouped by POS tags
		filtering out words with a log frequency greater than -8.0
		"""
		for word in sentence:
			if word.prob < -8.0 and numpy.count_nonzero(word.vector) > 0:  # filter all zero vectors
				if word.pos_ in pos_vector_dict:
					pos_vector_dict[word.pos_] = pos_vector_dict[word.pos_] + word.vector
				else:
					pos_vector_dict[word.pos_] = word.vector


class NoModelAvg(NoModelCBOW):
	def _group_by_pos(self, sentence, pos_vector_dict):
		""" Averaged word embeddings grouped by POS tags """
		pos_vecs = {}
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
