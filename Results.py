from __future__ import print_function

from datetime import datetime

class Result(object):
	"""
	Manage generating metrics for results and displaying them.
	At the moment only looking at binomial experiment i.e. solutions or not
	"""
	def __init__(self, guesses, solutions = None):
		self.datetime = datetime.now()
		self.guesses = guesses
		self.solutions = solutions
		self.mistakes = []

	def __str__(self):
		if self.solutions:
			assert len(self.solutions) == len(self.guesses)  # sanity check
			return "Experiment {}\nAccuracy: {}%".format(
				self.datetime.strftime("%Y-%m-%d at %H:%M:%S"),
				self.accuracy() * 100
			)
		else:
			return "No solutions provided. Can not calculate metrics."

	def accuracy(self):
		correct = 0
		size = len(self.solutions)
		for i in range(0, size):
			if self.guesses[i] == self.solutions[i]:
				correct += 1
			else:
				self.mistakes.append(self.guesses[i])
		return float(correct) / size

	def save_guesses(self, filename):
		print("Saving results to ", filename)
		with open(filename, 'w') as f:
			for g in self.guesses:
				f.write(g + '\n')

