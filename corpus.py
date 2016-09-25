from __future__ import print_function

import re
import random
import math
from collections import OrderedDict
from itertools import islice
from multiprocessing import Pool

import numpy
import spacy
from spacy import attrs

print("Loading English module...")
nlp = spacy.load('en')

DATA_DIR = 'challenge_data/'
TRAINING_DIALOGS_FILE = DATA_DIR+'train_dialogs.txt'
TRAINING_MISSING_FILE = DATA_DIR+'train_missing.txt'


model = None  # used by the multiprocessing functions
def pool_init(arg):
	""" initializer for sub-processes. """
	global model
	model = arg


class Formare(object):
	"""
	Formare, from the latin: to form
	Encapsulate formatting of a list of spaCy Docs.
	Good for dependency injection type designs.
	"""
	def __init__(self, attr=attrs.ID, max_length=0, pad_with=0):
		self.attr = [attr]
		self.max_length = max_length
		self.pad_with = pad_with

	def do(self, doc_list):
		"""
		Apply formating settings to a List of spaCy Docs
		:param doc_list: list of Docs
		:return: list of lists
		"""
		return Formare.conformare(self.transform(doc_list), length=self.max_length, value=self.pad_with)

	def transform(self, doc_list):
		""" Convert the docs into lists of one of the doc's attributes """
		return map(lambda x: x.to_array(self.attr).T[0], doc_list)

	@staticmethod
	def conformare(divergent_lists, length=0, value=0):
		"""
		Pad or crop the lengths of the sequences in the list passed. Make them conform to a specific length.
		:param divergent_lists: List of Lists to make conform
		:param length: int
		:param value: value to pad with
		:return: List of Lists with uniform length.
		"""
		if length == 0:
			return divergent_lists
		out = []
		for item in divergent_lists:
			if not type(item) == numpy.ndarray:
				item = numpy.array(item, dtype=numpy.int32)
			if len(item) > length:  # crop
				out += [item[:length]]
			elif len(item) < length:  # pad
				z = numpy.zeros(length, dtype=numpy.int32)
				z[-item.shape[0]:] = item
				out += [ z ]
			else:  # just right
				out += [item]
		return out

class BlobFormare(Formare):
	def do(self, doc_list):
		""" Merge the list of docs together before conformare """
		blob = numpy.concatenate(self.transform(doc_list))
		return Formare.conformare([blob], length=self.max_length, value=self.pad_with)[0]

class Corpus(object):
	"""
	Manage the corpus data.
	"""
	def __init__(self, dialogs_file=TRAINING_DIALOGS_FILE, missing_file=TRAINING_MISSING_FILE,
				 training=False, parse=False, tag=False, clip_data_at=0):

		self.nlp = nlp
		self.parse = parse
		self.training = training

		# Process Dilaogs file...

		if type(dialogs_file) == str:
			self.raw_dialogs  = Corpus.load_raw_data(dialogs_file)
		elif type(dialogs_file) in (dict, OrderedDict):
			self.raw_dialogs = dialogs_file
		else:
			raise ValueError('Nether a file path nor pre-digested corpus dialogs data was passed! Got "{}" instead?'.format(type(dialogs_file)))

		# ... and Missing utterances file ... 2nd verse, same as the first (almost).

		if not missing_file is None:
			if type(missing_file) == str:
				self.raw_missing = Corpus.load_raw_data(missing_file, True)
			elif type(missing_file) in (dict, OrderedDict):
				self.raw_missing = missing_file
			else:
				raise ValueError(
					'Nether a file path nor pre-digested corpus missing utterance data was passed! Got "{}" instead?'.format(
						type(missing_file)))
			assert len(self.raw_missing) == len(self.raw_dialogs)  # sanity check

		# For testing purposes only.
		if clip_data_at > 0:
			self.raw_missing = OrderedDict(islice(self.raw_missing.iteritems(), clip_data_at))
			self.raw_dialogs = OrderedDict(islice(self.raw_dialogs.iteritems(), clip_data_at))

		# Instantiate Dialog objects...
		# TODO: avoid duplication when calling .split()
		if training:
			dialogs = [(id, Dialog(id, d, self.raw_missing[id], parse, tag)) for id, d in self.raw_dialogs.iteritems()]
		else:
			dialogs = [(id, Dialog(id, d, None, parse, tag)) for id, d in self.raw_dialogs.iteritems()]

		self.dialogs = dict(dialogs)
		self.length = len(self.dialogs)

	def stats(self):
		""" Get some stats on the corpus dialogs and utterances. """
		def update_counts(n, tally):
			tally['count'] += n
			if n > tally['max']:
				tally['max'] = n
			if n < tally['min']:
				tally['min'] = n
		# Utterances length stats: min, max, mean .
		uttr = {'min': 9999, 'max': 0, 'count': 0, 'mean': 0}  # word count
		# Dialog utterances stats: min, max, mean.
		dlg = {'min': 9999, 'max': 0, 'count': 0, 'mean': 0}  # utterance count
		for _, d in self.dialogs.iteritems():
			update_counts(len(d), dlg)
			for u in d.all:
				update_counts(len(u), uttr)
		uttr['mean'] = uttr['count'] / dlg['count']
		dlg['mean'] = dlg['count'] / len(self.dialogs)
		return dlg, uttr

	@property
	def pool(self):
		""" Multi processor Pool instance for spreeding the load over all CPU cores. """
		# Always get a fresh worker pool initialized with the latest.
		return Pool(initializer = pool_init, initargs = (self, ))

	@staticmethod
	def load_raw_data(filename, as_single_sentences = False):
		""" Load, clean and shape raw data from the files. """
		print("Reading raw data file ", filename)
		with open(filename,'r') as f:
			return Corpus.clean_raw_text(f.readlines(), as_single_sentences)

	@staticmethod
	def clean_raw_text(raw_dialogs, one_sentence = False):
		"""
		Remove the cruff from the beginning of each line and organise text into dialogs
		indexed by ID. Parse with spaCy.
		"""
		print("Processing...")
		output = {}
		for sentence in raw_dialogs:
			id, text = sentence.strip('\n').split(' +++$+++ ')
			s = re.sub('<[^<]+?>', '', text) # clean out html tags
			s = re.sub(r'([.]+?)\1+', r'\1', s)  # collapse repeated '.....' to a single '.'
			en = unicode(s, "utf-8")
			if not one_sentence:
				if id in output:
					output[id] += [en]
				else:
					output[id] = [en]
			else:
				output[id] = en
		return OrderedDict(sorted(output.iteritems()))

	def sentence_triples_cbow(self, cv_split = 0.1):
		""" triples of sentences: previous, target & following sentences as CBOW dependency vectors 300 long each.
		:param cv_split: How much of the data to reserve for cross validation.
		:return: tuple (training, cross_validation) data sets.
		"""
		# data = self.pool.map(sentence_triples_cbow, self.dialogs.iteritems())
		data = map(lambda d: self._sentence_triples_cbow(d), self.dialogs.iteritems())
		# Flatten the complex nested arrays
		X = []; Y = []
		for dialog in data:
			Xs, Ys = dialog
			X += Xs; Y += Ys

		return Corpus.split_sets(X, Y, cv_split)

	@staticmethod
	def split_sets(X, Y, cv_split):
		X_train, X_test = Corpus.split_data(X, cv_split)
		Y_train, Y_test = Corpus.split_data(Y, cv_split)
		return (X_train, Y_train), (X_test, Y_test)

	@staticmethod
	def split_data(data, cv_split):
		sp = int(math.floor(len(data) * (1 - cv_split)))  # sp = split point
		return data[:sp], data[sp:]

	def split(self, cv_split = 0.1):
		"""
		Return two new Corpus' from splitting this one.
		:param cv_split:
		:return: Two Corpora, one for each side of the split i.e. training, testing
		"""
		split_point = int(math.floor(len(self.raw_dialogs) * (1 - cv_split)))
		dlgA, dlgB = Corpus.split_dict(self.raw_dialogs, split_point)
		misA, misB = Corpus.split_dict(self.raw_missing, split_point)
		A = Corpus(dlgA, misA, self.training, self.parse,self.parse)
		B = Corpus(dlgB, misB, self.training,self.parse,self.parse)
		return A, B

	@staticmethod
	def split_dict(dic, split_point):
		A = OrderedDict(islice(dic.iteritems(), split_point))
		B = OrderedDict(islice(dic.iteritems(), split_point, None))
		return A, B

	def _sentence_triples_cbow(self, (id, dialog)):
		X = []
		Y = []
		# for id, dialog in self.dialogs.iteritems():
		for triple in dialog.all_triples_vectors():
			X += [numpy.concatenate(triple)]  # positive case
			X += [numpy.concatenate((triple[0], self.random_utterance(not_id=id).vector, triple[2]))]  # negative case
			Y += [1, 0]
		return (X, Y)

	def fashion_missing(self, form=Formare()):
		keys, missing = zip(*[(key, d.missing) for key, d in self.dialogs.iteritems()])
		missing = form.do(missing)
		return dict(zip(*[keys, missing]))

	def fashion_dialogs(self, form=Formare()):
		"""
		Apply Formare to all the dialogs and return them as a dict
		:param form: Formare
		:return: dict
		"""
		return { key: form.do(d) for key, d in self.dialogs.iteritems() }

	def random_utterance(self, not_id):
		selected_id = not_id
		while selected_id == not_id:
			selected_id = random.choice(self.raw_missing.keys())
		return self.dialogs[selected_id].random_utterance()

	def all_dialog_missing_pair_as(self, attribute = attrs.ID, maxlen=28):
		"""
		All the utterances in the dialog as concatenate together and the missing
		utterence as a pair.
		Return AS lists of one of the spaCy Token attributes.
		"""
		d, m, Y = [], [], []  # dialogs, missing (with fakes), classes
		for id, dialog in self.dialogs.iteritems():
			d += [numpy.concatenate(Dialog.to_array(dialog.dialog, attribute))] * 2
			m += [Dialog.to_array([dialog.missing], attribute)[0]]
			m += [Dialog.to_array([self.random_utterance(id)], attribute)[0]]  # negative cases
			Y += [1,0]
		if maxlen > 0:
			d = Formare.conformare(d, maxlen * 5)
			m = Formare.conformare(m, maxlen)
		return d, m, Y

	def get_blobs(self, form=Formare()):
		return Formare.conformare([ d.blob(Formare(attr=form.attr[0])) for key, d in self.dialogs.iteritems() ], form.max_length)

	def get_sandwiches(self, form=Formare()):
		"""
		Triple of the utterances: before the missing, the missing, and after the missing.
		Oh yeah, we can make sandwiches ;)
		:return: dialog[-2] + missing + dialog[-1] for each dialog as a List
		"""
		return zip(*[ d.sandwich(form) for key, d in self.dialogs.iteritems()])

	def get_buns(self, form=Formare()):
		return zip(*[d.buns(form) for key, d in self.dialogs.iteritems()])

	def get_ham(self, form=Formare()):
		"""
		Formated Missing utterances (the bit between the buns :)
		"""
		return form.do([d.missing for key, d in self.dialogs.iteritems()])

	def get_spam(self, form=Formare()):
		"""
		List of random utterances for negative training cases
		:return: list of numpy.ndarray
		"""
		return form.do([self.random_utterance(not_id=key) for key, d in self.dialogs.iteritems()])

def sentence_triples_cbow((id, dialog)):
	""" multiprocessing pool version of this function """
	X = []
	Y = []
	for triple in dialog.all_triples_vectors():
		X += [numpy.concatenate(triple)]  # positive case
		if model.generate_negatives:
			# negative case
			X += [numpy.concatenate((triple[0], model.random_utterance(not_id=id).vector, triple[2]))]
			Y += [1, 0]
		else:
			Y += [1]
	return (X, Y)



class Dialog(object):
	"""
	Wrapper for the dialog data.
	Processes all the utterances with spaCy, optionally parsing them (slow).
	"""
	def __init__(self, id, dialog, missing = None, parse = False, tag = False):
		self.parse = parse
		self.tag = tag
		self.id = id
		self.raw_dialog = map(unicode, dialog)
		self.raw_missing = unicode(missing)
		self._all = None
		self._missing = None
		self._dialog = None

		# self.dialog = [nlp(utter, parse=parse, tag=tag) for utter in self.raw_dialog]

	def __str__(self):
		return "".join(["\n#",
			self.id,
			"\n", "\n".join(self.raw_dialog[:-1]),
			"\n>>> ", self.raw_missing,
			"\n", self.raw_dialog[-1:][0],
		])

	def __len__(self):
		return len(self.all)

	def __getitem__(self, key):
		""" Process the strings with spaCy in a lazy way, only when called for """
		if isinstance(key, slice):
			return [self.all[i] for i in xrange(*key.indices(len(self)))]
		elif isinstance(key, int):
			if key < 0 : #Handle negative indices
				key += len(self)
			if key < 0 or key >= len(self) :
				raise IndexError, "The index (%d) is out of range."%key
			return self.all[key]
		else:
			raise TypeError, "Invalid argument type."

	@property
	def missing(self):
		if not self._missing:
			self._missing = nlp(self.raw_missing, parse=self.parse, tag=self.tag, entity=False)
		return self._missing

	@property
	def dialog(self):
		if not self._dialog:
			self._dialog = [nlp(utter, parse=self.parse, tag=self.tag, entity=False) for utter in self.raw_dialog]
		return self._dialog

	@property
	def all(self):
		"""
		:return: List of spacy Doc instances
		"""
		if self.raw_missing:
			return self.dialog[:-1] + [self.missing] + self.dialog[-1:]
		else:
			return self.dialog

	def all_triples_vectors(self):
		"""
		:return: tuple of (previous, target, following) utterances for all sentences except the first and last
		"""
		return [ (self.all[i-1].vector, self.all[i].vector, self.all[i+1].vector) for i in range(1, len(self) - 1)]

	def random_utterance(self):
		return random.choice(self)

	def sandwich(self, form=Formare()):
		return form.do((self.dialog[-2], self.missing, self.dialog[-1]))

	def buns(self, form=Formare()):
		return form.do((self.dialog[-2], self.dialog[-1]))

	def blob(self, form=Formare()):
		return numpy.concatenate([doc.to_array(form.attr).T[0] for doc in self.dialog])

	@staticmethod
	def to_array(dialogs, attribute):
		return map(numpy.concatenate, [ utter.to_array([attribute]).T for utter in dialogs ])
