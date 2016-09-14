from multiprocessing import Pool

import spacy



experiment

def train(data):
	experiment.model(data)

class Experiment(object):
	def __init__(self, ModelClass, data):
		print "Loading English module..."
		self.model = ModelClass(spacy.load('en'))
		self.dialogs = data['dialogs']
		self.missing = data['missing']
		experiment = self

	def run():
		pool = Pool()
		pool.map(train, self.data.items())
