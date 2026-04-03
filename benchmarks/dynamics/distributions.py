import numpy as np


class GaussianDiagonalCov(dict):
	def __init__(self, cov_diag):
		cov_diag = np.asarray(cov_diag, dtype=float)
		super().__init__(
			mean=np.zeros(cov_diag.shape[0], dtype=float),
			cov=np.diag(cov_diag),
			cov_diag=cov_diag,
			stdev=np.sqrt(cov_diag),
		)

	def sample(self, size=None, rng=None):
		rng = np.random if rng is None else rng
		return rng.multivariate_normal(self['mean'], self['cov'], size=size)
