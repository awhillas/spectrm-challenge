import math
import random

import numpy
from spacy.parts_of_speech import IDS as POS_tags

random.seed(1234567890)  # for reproducibility

class GeneticAlgorithm(object):
	def __init__(self, max_pop_size = 10, keep = 0.2):
		super(GeneticAlgorithm, self).__init__()
		self.max_pop_size = max_pop_size
		self.keep = math.floor(max_pop_size * keep)
		self.population = []
		self.epochs = 10
		self.mutation_rate = 0.1  # how much of the population to mutate...
		self.mutation_amount = 0.7  # ...by how much
		self.SolutionClass = GASolution  # class to

	def run(self):
		def get_random(item_list, but_not):
			result = but_not
			while result != but_not:
				result = random.choice(item_list)
			return result

		elite = self.evaluate(self.initial_population())  # Choose the best of the initial population.

		for epoch in range(self.epochs):
			new_population = elite

			print("Crossover...")
			for i in range(self.max_pop_size - self.keep):  # make babies to fill up the rest of the population
				parentA = random.choice(elite)
				parentB = get_random(elite, parentA)
				new_population += parentA.crossover(parentB)

			print("Mutate...")
			for i in range(len(new_population)):
				if random.random() < self.mutation_rate:
					new_population[i].mutate(self.mutation_amount)

			print("Evaluate...")
			elite = self.evaluate(new_population)

		return max(elite, key=lambda item: item.score)

	def initial_population(self):
		return [self.SolutionClass() for _ in range(self.max_pop_size)]

	def evaluate(self, population):
		"""
			Survival of the fittest!
			Evaluate the entire population select the fittest.
		"""
		results = [solution.fitness(self.test_data()) for solution in population]
		scores = numpy.array(results)
		return [population[i] for i in numpy.argsort(scores)[-self.keep:]]

	def test_data(self):
		return None


class GASolution(object):
	def __init__(self, gen=None):
		super(GASolution, self).__init__()
		self.gen = gen if not gen is None else self.new_random_gen()
		self.score = 0

	@staticmethod
	def random_gen_value():
		return random.random() * 2 - 1

	@staticmethod
	def new_random_gen():
		return dict((pos, GASolution.random_gen_value()) for pos in POS_tags.keys())

	def mutate(self, amount = 0.1):
		"""
		Randomly mutate gen with given probability
		:param amount: chance of mutation
		:return: GASolution (mutant)
		"""
		mutant_gen = {}
		for key, value in self.gen:
			if random.random() < amount:
				mutant_gen[key] = GASolution.random_gen_value()
		return self.factory(mutant_gen)

	def crossover(self, partner, prob = 0.7):
		"""
		Perform uniform-crossover with the given partner.
		see: http://www.wseas.org/multimedia/journals/computers/2013/5705-156.pdf
		:param partner: GASolution
		:param prob: probability that the gens will be swaped
		:return: 2 x GASolution
		"""
		child_a = {}
		child_b = {}
		for key in self.gen.keys():
			if random.random() < prob:
				child_a[key] = partner.gen[key]
				child_b[key] = self.gen[key]
			else:
				child_a[key] = self.gen[key]
				child_b[key] = partner.gen[key]

		return self.factory(child_a), self.factory(child_b)

	@staticmethod
	def factory(gen=None):
		""" So we can override this in the child class """
		return GASolution(gen)

	def fitness(self, test_data=None):
		""" Return fitness score. Override with problem specific fitness value. """
		pass
