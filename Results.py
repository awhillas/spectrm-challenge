from datetime import datetime

class Result(object):
	"""
	Manage generating mettrics for results and displaying them.
	At the moment only looking at binomial experiment i.e. solutions or not
	"""
	def __init__(self, guesses, solutions = None):
		self.datetime = datetime.now()
		self.guesses = guesses
		self.solutions = solutions
		self.mistakes = []

	def precision(self):
		correct = 0
		size = len(self.solutions)
		for i in range(0, size):
			if self.guesses[i] == self.solutions[i]:
				correct += 1
			else:
				self.mistakes.append(self.guesses[i])
		return float(correct) / size

	def correct(self):
		return []

	def __str__(self):
		if self.solutions:
			assert len(self.solutions) == len(self.guesses)
			return "Experiment {}\nPrecision: {}%".format(
				self.datetime.strftime("%Y-%m-%d at %H:%M:%S"),
				self.precision() * 100
			)
		else:
			return "No solutions provided. Can not calculate metrics."
