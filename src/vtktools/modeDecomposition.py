from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

class ModeDecomposer:
	class PODMode:
		def __init__(self, parent, modeId):
			self.id = modeId
			self.mode = parent.invWeights * parent.u[:, modeId]
			self.amplitude = parent.s[modeId]
			self.timeCoeffs = self.amplitude * parent.v[modeId, :]

		def reconstruct(self):
			return np.outer(self.mode, self.timeCoeffs)

	class DMDMode:
		def __init__(self, parent, modeId):
			self.id = modeId
			eigvec = parent.S_eigvecs[:, modeId]
			self.mode = np.dot(parent.invWeights[:, np.newaxis] * parent.u[:, :eigvec.shape[0]], eigvec)
			self.eigenvalue = parent.S_eigvals[modeId]
			self.growthRate = np.log(np.abs(self.eigenvalue))
			self.frequency = np.arctan2(np.imag(self.eigenvalue), np.real(self.eigenvalue))

	def __init__(self):
		''' Construct a mode analyser
		'''

		# Data
		self.x = None
		self.weights = None
		self.mean = None
		
		# SVD stuff
		self.svdTol = 1e-4
		self.u = None	# Left singular vectors
		self.v = None	# Right singular vectors
		self.s = None	# Singular values

		# DMD 
		self.S_eigvals = None
		self.S_eigvecs = None

	def compute(self, x, mass=None):
		'''
		Compute the POD and DMD modes of the dataset
			:x: Data series with each column corresponding to one timestep
			:mass: Vector of positive weights for the Singular value decomposition
		'''
		# No mass vector provided, assume equal weighting
		if mass is None:
			self.weights = np.ones(x.shape[0])
		else:
			if mass.ndim != 1:
				raise ValueError('The mass must be a vector!')
			if mass.shape[0] != x.shape[0]:
				raise ValueError('The number of mass elements did not match the number of rows in the data')
			self.weights = np.sqrt(mass)

		self.invWeights = 1. / self.weights
		
		# Subtract mean
		self.x = x
		self.mean = np.mean(self.x, axis=1)
		self.x -= self.mean[:, np.newaxis]

		# Compute
		self._computeSvd()
		self._computeDMD()

	def _computeSvd(self):
		self.u, self.s, self.v = np.linalg.svd(self.weights[:, np.newaxis] * self.x[:, 0:-1], full_matrices = False)

	def _computeDMD(self):
		# Evaluate inverse of the singular values
		sigmaInv = 1. / self.s[self.s > self.svdTol]
		nSV = sigmaInv.size	# Number of non-zeros singular values

		# Evaluate pseudo-companion matrix
		S = np.dot( \
				np.dot( \
					(self.weights[:, np.newaxis] * self.u[:, :nSV]).conj().T, \
					self.x[:, 1:]), \
				self.v[:nSV, :].conj().T) \
			* sigmaInv[np.newaxis, :]

		# Compute eigenvalues
		eigvals, eigvecs = np.linalg.eig(S)

		# Sort eigenvalues in descending order
		inds = np.flipud(np.argsort(eigvals))
		self.S_eigvals = eigvals[inds]
		self.S_eigvecs = eigvecs[:, inds]

	def getPODMode(self, modeId):
		self.assertComputed()
		return ModeDecomposer.PODMode(self, modeId)

	def assertComputed(self):
		if not self.hasComputed():
			raise RuntimeError('The decomposition has not been computed, run compute() before accessing the mode data')

	def hasComputed(self):
		return not self.u is None

	def getPODTimeCoeffs(self, maxModeId=-1):
		if maxModeId == 1:
			return self.s[:, np.newaxis] * self.v
		else:
			return np.row_stack((
				self.s[:maxModeId, np.newaxis] * self.v[:maxModeId, :],
				[np.sum(self.s[maxModeId:, np.newaxis] * self.v[maxModeId:, :], axis=0)]))

	def getDMDMode(self, modeId):
		self.assertComputed()
		return ModeDecomposer.DMDMode(self, modeId)

	def getDMDEigenvalues(self):
		return self.S_eigvals

	def reconstructFromPOD(self, numberOfModes):
		self.assertComputed()
		ret = np.zeros((self.x.shape[0], self.x.shape[1]-1))
		ret += self.mean[:, np.newaxis]
		for i in range(numberOfModes):
			ret += self.getPODMode(i).reconstruct()
		return ret

	def saveResults(self, fileName):
		self.assertComputed()
		np.savez(fileName, u = self.u, v = self.v, s = self.s, 
			weights = self.weights, invWeights = self.invWeights, svdTol = self.svdTol, 
			S_eigvals = self.S_eigvals, S_eigvecs = self.S_eigvecs,
			mean = self.mean)

	def loadResults(self, fileName):
		npFile = np.load(fileName)
		self.u = npFile['u']
		self.v = npFile['v'] 
		self.s = npFile['s']
		self.weights = npFile['weights']
		self.invWeights = npFile['invWeights']
		self.svdTol = npFile['svdTol']
		self.S_eigvals = npFile['S_eigvals']
		self.S_eigvecs = npFile['S_eigvecs']
		self.mean = npFile['mean']

if __name__ == '__main__':
	# Create sample data
	t = np.linspace(0, 40, 3000)
	x = np.linspace(0, 1, 50)
	f1 = np.sin(2*np.pi*x)
	f2 = np.cos(6*np.pi*x)
	f3 = np.sin(11*np.pi*x)
	data = 10 + 4*np.outer(f1, np.sin(4*np.pi*t)) + 2*np.outer(f2, np.sin(np.pi*t)) + np.outer(f3, np.sin(2*np.pi*t))

	# Create decomposer
	modeAnalyzer = ModeDecomposer()
	modeAnalyzer.compute(data)

	fig, axes = plt.subplots(3, 1)

	print(np.sum(np.abs(data[:, :-1] - (modeAnalyzer.getPODMode(0).reconstruct() + modeAnalyzer.getPODMode(1).reconstruct()))))

	for i in range(0, 5):
		axes[0].plot(x, modeAnalyzer.getPODMode(i).mode, label='Mode {:d}'.format(i))
		axes[1].plot(t[:-1], modeAnalyzer.getPODMode(i).timeCoeffs, label='Mode {:d}'.format(i))

	axes[0].set_xlabel('x')
	axes[1].set_xlabel('t')

	for i in range(0, 5):
		axes[2].plot(modeAnalyzer.getDMDMode(i).mode, label='Mode {:d}'.format(i))
	plt.legend()

	# Reconstruct and compare
	f2, ax2 = plt.subplots(2, 1)
	ax2[0].imshow(data, extent=[t[0], t[-1], x[0], x[-1]], aspect=10)
	ax2[1].imshow(modeAnalyzer.reconstructFromPOD(1), extent=[t[0], t[-1], x[0], x[-1]], aspect=10)

	plt.show()
