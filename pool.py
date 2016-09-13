from multiprocessing import Pool

PROCESSES = 4

t = None

def run(n):
	return t.f(n)

class Test(object):
	def __init__(self, number):
		self.number = number

	def f(self, x):
		print x * self.number

	def pool(self):
		pool = Pool(2)
		pool.map(run, range(10))

if __name__ == '__main__':
	global t
	t = Test(9)
	t.pool()
	pool = Pool(2)
	pool.map(run, range(10))
