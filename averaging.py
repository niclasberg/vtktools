import numpy as np

class NonUniformStatistics:
	def __init__(self, arrayShape):
		self.oldMean = np.zeros(arrayShape)
		self.newMean = np.zeros(arrayShape)
		self.sampleCount = np.zeros(arrayShape)
		self.std = np.zeros(arrayShape)
		self.max = np.zeros(arrayShape)

	def addValue(self, indx, value):
		self.sampleCount[indx] += 1
		if self.sampleCount[indx] == 1:
			self.oldMean[indx] = value
			self.newMean[indx] = value
		else:
			self.oldMean[indx] = self.newMean[indx]
			self.newMean[indx] += (value - self.oldMean[indx]) / self.sampleCount[indx]
			self.std[indx] += (value - self.oldMean[indx]) * (value - self.newMean[indx])
		self.max[indx] = max(self.max[indx], value)

	def getMean(self):
		return self.newMean

	def getStd(self):
		ret = np.zeros_like(self.std)
		inds = self.sampleCount >= 2
		ret[inds] = np.sqrt(self.std[inds] / (self.sampleCount[inds] - 1))
		return ret

	def getMax(self):
		return self.max

class RunningAverage:
	def __init__(self):
		self.mean = None
		self.m2 = None
		self.N = 0

	# Add a value
	def addValue(self, x):
		self.N += 1
		if self.N == 1:
			# First added value
			self.mean = np.copy(x)
			self.m2 = np.zeros_like(x)
		else:
			delta = x - self.mean
			self.mean += delta / float(self.N)
			self.m2 += delta * (x - self.mean)

	# Add multiple values
	# This method uses vectorized computations to speed things up
	def addValues(self, xs):
		# Compute average of the provided values
		N = len(xs)
		if N < 1:
			pass	# No values to add
		elif N == 1:
			self.addValue(xs)	# Only one value, default to the addValue method
		else:
			# General case. Compute the mean and 2nd moment of the supplied values
			self.addValuesByMeanAndVariance(np.mean(xs, axis=0), np.var(xs, axis=0), N)

	def addValuesByMeanAndVariance(self, xMean, varVal, N):
		xM2 = varVal * N
		if self.N == 0:		
			# No values have previously been added, just set the mean and the 
			# 2nd moment to those of the supplied values
			self.mean = xMean
			self.m2 = xM2
		else:
			# Merge the statistics for the two sets (Chan et al.)
			# See the wiki page on "algorithms for calculating variance" for more details
			self.m2 += xM2 + (xMean - self.mean)**2 * (self.N * N) / (self.N + N)
			self.mean = (self.N*self.mean + N*xMean) / (self.N + N)
		self.N += N

	def getMean(self):
		if self.N < 1:
			raise ValueError('At least one sample needed for mean value')
		return self.mean

	def getVariance(self):
		if self.N < 2:
			raise ValueError('At least two samples needed for variance')
		return self.m2 / self.N

	def getStd(self):
		return np.sqrt(self.N * self.getVariance() / (self.N-1.))

	def getRms(self):
		return np.sqrt(self.getVariance())

# Test the averaging
def _test():
	Nsamples = 10000
	x = np.random.rand(Nsamples)
	xMean = np.mean(x)
	xRms = np.std(x)
	
	# Compute average by inserting the values one by one
	av0 = RunningAverage()
	for xx in x:
		av0.addValue(xx)

	# Compute by splitting up the data into two chunks and adding
	av1 = RunningAverage()
	splitInd = np.random.randint(Nsamples)
	av1.addValues(x[:splitInd])
	av1.addValues(x[splitInd:])
	
	print 'Mean: np =', xMean, 'running =', av0.getMean(), 'running (chunked) =', av1.getMean()
	print 'RMS: np =', xRms, 'running =', av0.getRms(), 'running (chunked) =', av1.getRms()
	print 'Std: np =', np.sqrt(float(Nsamples)/(Nsamples-1.))*xRms, 'running =', av0.getStd(), 'running (chunked) =', av1.getStd()

if __name__ == '__main__':
	_test()
