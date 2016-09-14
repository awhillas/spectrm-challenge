from multiprocessing import Pool

t = None

def run(n):
	return t.f(n)

class Test(object):
	def __init__(self, number):
		self.number = number

	def f(self, x):
		return x * self.number

	def pool(self):
		pool = Pool()
		return pool.map(run, range(10))

if __name__ == '__main__':
	# global t
	t = Test(9)
	print t.pool()

	pool = Pool()
	result = pool.map(run, range(10))
	print result
