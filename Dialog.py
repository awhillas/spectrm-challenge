class Dialog(object):
	"""
	Nice wrapper for the data
	"""
	def __init__(self, id, data):
		self.id = id
		self._raw_dialog = data['dialogs'][id]
		self._raw_missing = data['missing'][id]
		self._dialog = [nlp(utter) for utter in self._raw_dialog]
		self._missing = nlp(self._raw_missing)

	def __str__(self):
		return "".join(["\n#",
			self.id,
			"\n", "\n".join(self._raw_dialog[:-1]),
			"\n>>> ", self._raw_missing,
			"\n", self._raw_dialog[-1:][0],
		])

	@property
	def dialog(self):
		return self._dialog

	@property
	def missing(self):
		return self._missing

	def display(self):
		out = ["\t".join((w.lemma_, w.pos_, w.dep_, str(w.prob)))
			for sentence in self.dialog
			for w in sentence
			if w.prob < -9.0
		]
		return out + ["--------------------------------"] + \
			["\t".join((w.lemma_, w.pos_, w.dep_, str(w.prob))) for w in self.missing]

	def by_pos(self, pos, prob = 0.0):
		""" Returns a list of tokens with the given POS tag and optionally a bellow a log probability """
		return [ word for sentence in self.dialog for word in sentence if word.pos == pos and word.prob < prob ]

	def by_prob(self, prob):
		return [ word for sentence in self.dialog for word in sentence if word.prob < prob ]
