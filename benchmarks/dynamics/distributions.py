import numpy as np
import jax
import jax.numpy as jnp
import logging

logger = logging.getLogger(__name__)

def _integ_Gauss(x_lb, x_ub, x, stdev):
	'''
	Integrate a univariate Gaussian distribution over a given interval.

	:param x_lb: Lower bound of the interval.
	:param x_ub: Upper bound of the interval.
	:param x: Mean of the Gaussian distribution.
	:param stdev: Standard deviation of the Gaussian distribution.
	:return:
	'''
	eps = 1e-6  # Add tiny epsilon to avoid NaN problems if the Gaussian is a Dirac (i.e., stdev=0) and x_lb or x_ub equals x
	return jax.scipy.stats.norm.cdf(x_ub, x + eps, stdev) - jax.scipy.stats.norm.cdf(x_lb, x + eps, stdev)


# vmap to compute multivariate Gaussian integral in n dimensions
_vmap_integ_Gauss = jax.jit(jax.vmap(_integ_Gauss, in_axes=(0, 0, 0, 0), out_axes=0))
_vmap_integ_Gauss_per_dim = jax.jit(jax.vmap(_integ_Gauss, in_axes=(0, 0, 0, None), out_axes=0))
_vmap_integ_Gauss_per_dim_single = jax.jit(jax.vmap(_integ_Gauss, in_axes=(0, 0, None, None), out_axes=0))


def _cdf_Triangular(x, mean, halfwidth):
	'''
	CDF of a symmetric triangular distribution with support [mean-halfwidth, mean+halfwidth].
	'''
	eps = 1e-8  # Prevent division by zero when halfwidth is (close to) zero.
	h = jnp.maximum(halfwidth, eps)
	a = mean - h
	b = mean + h

	left = ((x - a) ** 2) / (2 * h ** 2)
	right = 1 - ((b - x) ** 2) / (2 * h ** 2)

	return jnp.where(
		x <= a,
		0.0,
		jnp.where(x <= mean, left, jnp.where(x <= b, right, 1.0))
	)


def _integ_Triangular(x_lb, x_ub, x, halfwidth):
	'''
	Integrate a univariate symmetric triangular distribution over a given interval.
	'''
	prob = _cdf_Triangular(x_ub, x, halfwidth) - _cdf_Triangular(x_lb, x, halfwidth)
	return jnp.clip(prob, 0.0, 1.0)


def _closest_to_mean(x_lb, x_ub, mean_lb, mean_ub):
	'''
	Point in [mean_lb, mean_ub] that is closest to the interval midpoint.
	'''
	mean = (x_lb + x_ub) / 2
	return jnp.maximum(jnp.minimum(mean_ub, mean), mean_lb)


_vmap_integ_Triangular = jax.jit(jax.vmap(_integ_Triangular, in_axes=(0, 0, 0, 0), out_axes=0))
_vmap_integ_Triangular_per_dim = jax.jit(jax.vmap(_integ_Triangular, in_axes=(0, 0, 0, None), out_axes=0))
_vmap_integ_Triangular_per_dim_single = jax.jit(jax.vmap(_integ_Triangular, in_axes=(0, 0, None, None), out_axes=0))


class GaussianDistr(dict):
	def __init__(self, cov_diag):
		cov_diag = jnp.asarray(cov_diag, dtype=float)
		super().__init__(
			type='Gaussian',
			mean=jnp.zeros(cov_diag.shape[0], dtype=float),
			cov=jnp.diag(cov_diag),
			cov_diag=cov_diag,
			stdev=jnp.sqrt(cov_diag),
			support_radius=4 * jnp.sqrt(cov_diag), # 3 standard deviations cover 99.9% of the probability mass for a univariate Gaussian
		)

	def sample(self, size=None, rng=None):
		rng = np.random if rng is None else rng
		return rng.multivariate_normal(self['mean'], self['cov'], size=size)

	def sample_jax(self, rng, shape=()):
		mean = jnp.asarray(self['mean'])
		stdev = jnp.asarray(self['stdev'])
		if isinstance(shape, int):
			shape = (shape,)
		sample_shape = tuple(shape) + (mean.shape[0],)
		return mean + stdev * jax.random.normal(rng, shape=sample_shape)

	def set_partition_probs(self, num_cells):
		'''
		Partition the truncated support into a grid of cells and compute the joint
		probability mass of each cell.

		Each dimension i is divided into num_cells[i] equal-width intervals within
		[mean[i] - support_radius[i], mean[i] + support_radius[i]] (3 standard deviations).
		The full grid contains prod(num_cells) cells. Joint probabilities are the product
		of per-dimension probabilities (independent dimensions).
		The probability mass outside the truncated support is stored as the remainder.

		:param num_cells: Array of integers of length n_dims, number of cells per dimension.
		:return:
			- cells:     array of shape (prod(num_cells), n_dims, 2) with [lb, ub] per dim per cell.
			- probs:     array of shape (prod(num_cells),) with joint probability mass per cell.
			- remainder: scalar, total probability mass not captured by any cell.
		'''

		print(f'- Define noise partition with {num_cells} cells')

		mean = self['mean']
		stdev = self['stdev']
		support_radius = self['support_radius']
		n_dims = len(stdev)
		has_noise = np.asarray(stdev) > 0

		per_dim_cells = []
		per_dim_probs = []
		actual_num_cells = []

		for i in range(n_dims):
			if has_noise[i]:
				edges = jnp.linspace(mean[i] - support_radius[i], mean[i] + support_radius[i], num_cells[i] + 1)
				lbs = edges[:-1]
				ubs = edges[1:]
				p = _vmap_integ_Gauss_per_dim_single(lbs, ubs, mean[i], stdev[i])
				actual_num_cells.append(num_cells[i])
			else:
				lbs = jnp.array([mean[i]])
				ubs = jnp.array([mean[i]])
				p = jnp.array([1.0])
				actual_num_cells.append(1)
			per_dim_cells.append(jnp.stack([lbs, ubs], axis=1))
			per_dim_probs.append(p)

		# Build flat index arrays for the Cartesian product
		index_grids = np.meshgrid(*[np.arange(n) for n in actual_num_cells], indexing='ij')
		flat_idx = [g.reshape(-1) for g in index_grids]

		# cells[j, i, :] = per-dim bounds of dimension i for cell j
		all_cells = jnp.stack([per_dim_cells[i][flat_idx[i]] for i in range(n_dims)], axis=1)

		# joint probability = product of per-dimension probabilities
		all_probs = per_dim_probs[0][flat_idx[0]]
		for i in range(1, n_dims):
			all_probs = all_probs * per_dim_probs[i][flat_idx[i]]

		self.partition = {
			"cells": all_cells,
			"probs": all_probs,
			"remainder": 1.0 - jnp.sum(all_probs)
		}

		logger.info(
			'- Noise partition: %d cells total  |  remainder=%.4f',
			len(all_probs),
			float(1.0 - jnp.sum(all_probs)),
		)

		return 
	
	def prob_minmax(self, x_lb, x_ub, mean_lb, mean_ub, wrap_array):
		'''
		Compute the min/max integral of a multivariate Gaussian distribution over a given interval, where the mean of the Gaussian lies in [mean_lb, mean_ub].

		:param x_lb: Lower bound of the interval.
		:param x_ub: Upper bound of the interval.
		:param mean_lb: Lower bound of the mean of the Gaussian distribution.
		:param mean_ub: Upper bound of the mean of the Gaussian distribution.
		:param wrap_array: Wrap at the indices where this array is True.
		:return: Min/max probabilities.
		'''

		# Determine point closest to mean of region over which to integrate
		closest_to_mean = _closest_to_mean(x_lb, x_ub, mean_lb, mean_ub)

		# Maximum probability is the product
		p_max = jnp.prod(_vmap_integ_Gauss(x_lb, x_ub, closest_to_mean, self['stdev']) * ~wrap_array + 1 * wrap_array)

		p1 = _vmap_integ_Gauss(x_lb, x_ub, mean_lb, self['stdev']) * ~wrap_array + 1 * wrap_array
		p2 = _vmap_integ_Gauss(x_lb, x_ub, mean_ub, self['stdev']) * ~wrap_array + 1 * wrap_array
		p_min = jnp.prod(jnp.minimum(p1, p2))

		return jnp.array([p_min, p_max])
	
	def prob_minmax_per_dim(self, n, wrap, x_lb_per_dim, x_ub_per_dim, mean_lb, mean_ub, state_space_size):
		'''
		Compute the min/max integral of a multivariate Gaussian distribution over a given interval, where the mean of the Gaussian lies in [mean_lb, mean_ub].
		Exploit rectangular partition to compute much fewer Gaussian integrals

		:param n: Dimension of the state space.
		:param wrap: Wrap at the indices where this array is True.
		:param x_lb_per_dim: Lower bound of the interval per dimension.
		:param x_ub_per_dim: Upper bound of the interval per dimension.
		:param mean_lb: Lower bound of the mean of the Gaussian distribution.
		:param mean_ub: Upper bound of the mean of the Gaussian distribution.
		:param state_space_size: Size of the state space per dimension (i.e., size of the "state space box")
		:return:
			- Min/max probabilities.
			- Lower bound probabilities.
			- Upper bound probabilities.
		'''

		probs = []
		prob_low = []
		prob_high = []

		for i in range(n):
			x_lb = x_lb_per_dim[i]
			x_ub = x_ub_per_dim[i]
			closest_to_mean = _closest_to_mean(x_lb, x_ub, mean_lb[i], mean_ub[i])

			if wrap[i]:
				# Compute shifts without building intermediate array
				shift_neg = -state_space_size[i]
				shift_pos = state_space_size[i]

				res_neg = _vmap_integ_Gauss_per_dim(
					x_lb + shift_neg,
					x_ub + shift_neg,
					_closest_to_mean(x_lb + shift_neg, x_ub + shift_neg, mean_lb[i], mean_ub[i]),
					self['stdev'][i])
				res_zero = _vmap_integ_Gauss_per_dim(
					x_lb,
					x_ub,
					closest_to_mean,
					self['stdev'][i])
				res_pos = _vmap_integ_Gauss_per_dim(
					x_lb + shift_pos,
					x_ub + shift_pos,
					_closest_to_mean(x_lb + shift_pos, x_ub + shift_pos, mean_lb[i], mean_ub[i]),
					self['stdev'][i])
				p_max = res_neg + res_zero + res_pos

				res1_neg = _vmap_integ_Gauss_per_dim_single(x_lb + shift_neg, x_ub + shift_neg, mean_lb[i], self['stdev'][i])
				res2_neg = _vmap_integ_Gauss_per_dim_single(x_lb + shift_neg, x_ub + shift_neg, mean_ub[i], self['stdev'][i])
				res1_zero = _vmap_integ_Gauss_per_dim_single(x_lb, x_ub, mean_lb[i], self['stdev'][i])
				res2_zero = _vmap_integ_Gauss_per_dim_single(x_lb, x_ub, mean_ub[i], self['stdev'][i])
				res1_pos = _vmap_integ_Gauss_per_dim_single(x_lb + shift_pos, x_ub + shift_pos, mean_lb[i], self['stdev'][i])
				res2_pos = _vmap_integ_Gauss_per_dim_single(x_lb + shift_pos, x_ub + shift_pos, mean_ub[i], self['stdev'][i])

				p_min = jnp.minimum(res1_neg, res2_neg) + jnp.minimum(res1_zero, res2_zero) + jnp.minimum(res1_pos, res2_pos)
			else:
				p_max = _vmap_integ_Gauss_per_dim(x_lb, x_ub, closest_to_mean, self['stdev'][i])
				p_min = jnp.minimum(_vmap_integ_Gauss_per_dim_single(x_lb, x_ub, mean_lb[i], self['stdev'][i]),
									_vmap_integ_Gauss_per_dim_single(x_lb, x_ub, mean_ub[i], self['stdev'][i]))

			probs.append(jnp.vstack([p_min, p_max]).T)
			prob_low.append(p_min)
			prob_high.append(p_max)

		return probs, prob_low, prob_high


class TriangularDistr(dict):
	def __init__(self, halfwidth):
		halfwidth = jnp.asarray(halfwidth, dtype=float)
		super().__init__(
			type='Triangular',
			mean=jnp.zeros(halfwidth.shape[0], dtype=float),
			halfwidth=halfwidth,
			support_radius=halfwidth, # Support radius of triangular distribution is equal to halfwidth
		)
		
	def sample(self, size=None, rng=None):
		rng = np.random if rng is None else rng
		mean = np.asarray(self['mean'])
		halfwidth = np.asarray(self['halfwidth'])
		nonzero = halfwidth > 0
		halfwidth_safe = np.where(nonzero, halfwidth, 1.0)

		if size is None:
			sample_size = mean.shape
		elif np.isscalar(size):
			sample_size = (int(size), mean.shape[0])
		else:
			sample_size = tuple(size) + (mean.shape[0],)

		sampled = rng.triangular(mean - halfwidth_safe, mean, mean + halfwidth_safe, size=sample_size)
		return np.where(nonzero, sampled, mean)

	def sample_jax(self, rng, shape=()):
		mean = jnp.asarray(self['mean'])
		halfwidth = jnp.asarray(self['halfwidth'])
		if isinstance(shape, int):
			shape = (shape,)
		sample_shape = tuple(shape) + (mean.shape[0],)
		rng1, rng2 = jax.random.split(rng)
		u1 = jax.random.uniform(rng1, shape=sample_shape)
		u2 = jax.random.uniform(rng2, shape=sample_shape)
		return mean + halfwidth * (u1 - u2)

	def set_partition_probs(self, num_cells):
		'''
		Partition the support into a grid of cells and compute the joint probability
		mass of each cell.

		Each dimension i is divided into num_cells[i] equal-width intervals within
		[mean[i] - halfwidth[i], mean[i] + halfwidth[i]]. The full grid contains
		prod(num_cells) cells. Joint probabilities are the product of per-dimension
		probabilities (independent dimensions).

		:param num_cells: Array of integers of length n_dims, number of cells per dimension.
		:return:
			- cells:     array of shape (prod(num_cells), n_dims, 2) with [lb, ub] per dim per cell.
			- probs:     array of shape (prod(num_cells),) with joint probability mass per cell.
			- remainder: scalar, total probability mass not captured by any cell
			             (theoretically zero; captures floating-point residuals).
		'''
		mean = self['mean']
		halfwidth = self['halfwidth']
		n_dims = len(halfwidth)
		has_noise = np.asarray(halfwidth) > 0

		per_dim_cells = []
		per_dim_probs = []
		actual_num_cells = []

		for i in range(n_dims):
			if has_noise[i]:
				edges = jnp.linspace(mean[i] - halfwidth[i], mean[i] + halfwidth[i], num_cells[i] + 1)
				lbs = edges[:-1]
				ubs = edges[1:]
				p = _vmap_integ_Triangular_per_dim_single(lbs, ubs, mean[i], halfwidth[i])
				actual_num_cells.append(num_cells[i])
			else:
				lbs = jnp.array([mean[i]])
				ubs = jnp.array([mean[i]])
				p = jnp.array([1.0])
				actual_num_cells.append(1)
			per_dim_cells.append(jnp.stack([lbs, ubs], axis=1))
			per_dim_probs.append(p)

		# Build flat index arrays for the Cartesian product
		index_grids = np.meshgrid(*[np.arange(n) for n in actual_num_cells], indexing='ij')
		flat_idx = [g.reshape(-1) for g in index_grids]

		# cells[j, i, :] = per-dim bounds of dimension i for cell j
		all_cells = jnp.stack([per_dim_cells[i][flat_idx[i]] for i in range(n_dims)], axis=1)

		# joint probability = product of per-dimension probabilities
		all_probs = per_dim_probs[0][flat_idx[0]]
		for i in range(1, n_dims):
			all_probs = all_probs * per_dim_probs[i][flat_idx[i]]

		self.partition = {
			"cells": all_cells,
			"probs": all_probs,
			"remainder": 1.0 - jnp.sum(all_probs)
		}

		return

	def prob_minmax(self, x_lb, x_ub, mean_lb, mean_ub, wrap_array):
		'''
		Compute the min/max integral of a multivariate triangular distribution over a given interval,
		where the mean of the triangular distribution lies in [mean_lb, mean_ub].

		:param x_lb: Lower bound of the interval.
		:param x_ub: Upper bound of the interval.
		:param mean_lb: Lower bound of the mean of the triangular distribution.
		:param mean_ub: Upper bound of the mean of the triangular distribution.
		:param wrap_array: Wrap at the indices where this array is True.
		:return: Min/max probabilities.
		'''

		# Determine point closest to mean of region over which to integrate.
		closest_to_mean = _closest_to_mean(x_lb, x_ub, mean_lb, mean_ub)

		# Maximum probability is the product.
		p_max = jnp.prod(_vmap_integ_Triangular(x_lb, x_ub, closest_to_mean, self['halfwidth']) * ~wrap_array + 1 * wrap_array)

		p1 = _vmap_integ_Triangular(x_lb, x_ub, mean_lb, self['halfwidth']) * ~wrap_array + 1 * wrap_array
		p2 = _vmap_integ_Triangular(x_lb, x_ub, mean_ub, self['halfwidth']) * ~wrap_array + 1 * wrap_array
		p_min = jnp.prod(jnp.minimum(p1, p2))

		return jnp.array([p_min, p_max])

	def prob_minmax_per_dim(self, n, wrap, x_lb_per_dim, x_ub_per_dim, mean_lb, mean_ub, state_space_size):
		'''
		Compute the min/max integral of a multivariate triangular distribution over a given interval,
		where the mean of the triangular distribution lies in [mean_lb, mean_ub].
		Exploit rectangular partition to compute much fewer triangular integrals.

		:param n: Dimension of the state space.
		:param wrap: Wrap at the indices where this array is True.
		:param x_lb_per_dim: Lower bound of the interval per dimension.
		:param x_ub_per_dim: Upper bound of the interval per dimension.
		:param mean_lb: Lower bound of the mean of the triangular distribution.
		:param mean_ub: Upper bound of the mean of the triangular distribution.
		:param state_space_size: Size of the state space per dimension (i.e., size of the "state space box")
		:return:
			- Min/max probabilities.
			- Lower bound probabilities.
			- Upper bound probabilities.
		'''

		probs = []
		prob_low = []
		prob_high = []

		for i in range(n):
			x_lb = x_lb_per_dim[i]
			x_ub = x_ub_per_dim[i]
			closest_to_mean = _closest_to_mean(x_lb, x_ub, mean_lb[i], mean_ub[i])

			if wrap[i]:
				shift_neg = -state_space_size[i]
				shift_pos = state_space_size[i]

				res_neg = _vmap_integ_Triangular_per_dim(
					x_lb + shift_neg,
					x_ub + shift_neg,
					_closest_to_mean(x_lb + shift_neg, x_ub + shift_neg, mean_lb[i], mean_ub[i]),
					self['halfwidth'][i])
				res_zero = _vmap_integ_Triangular_per_dim(
					x_lb,
					x_ub,
					closest_to_mean,
					self['halfwidth'][i])
				res_pos = _vmap_integ_Triangular_per_dim(
					x_lb + shift_pos,
					x_ub + shift_pos,
					_closest_to_mean(x_lb + shift_pos, x_ub + shift_pos, mean_lb[i], mean_ub[i]),
					self['halfwidth'][i])
				p_max = res_neg + res_zero + res_pos

				res1_neg = _vmap_integ_Triangular_per_dim_single(x_lb + shift_neg, x_ub + shift_neg, mean_lb[i], self['halfwidth'][i])
				res2_neg = _vmap_integ_Triangular_per_dim_single(x_lb + shift_neg, x_ub + shift_neg, mean_ub[i], self['halfwidth'][i])
				res1_zero = _vmap_integ_Triangular_per_dim_single(x_lb, x_ub, mean_lb[i], self['halfwidth'][i])
				res2_zero = _vmap_integ_Triangular_per_dim_single(x_lb, x_ub, mean_ub[i], self['halfwidth'][i])
				res1_pos = _vmap_integ_Triangular_per_dim_single(x_lb + shift_pos, x_ub + shift_pos, mean_lb[i], self['halfwidth'][i])
				res2_pos = _vmap_integ_Triangular_per_dim_single(x_lb + shift_pos, x_ub + shift_pos, mean_ub[i], self['halfwidth'][i])

				p_min = jnp.minimum(res1_neg, res2_neg) + jnp.minimum(res1_zero, res2_zero) + jnp.minimum(res1_pos, res2_pos)
			else:
				p_max = _vmap_integ_Triangular_per_dim(x_lb, x_ub, closest_to_mean, self['halfwidth'][i])
				p_min = jnp.minimum(_vmap_integ_Triangular_per_dim_single(x_lb, x_ub, mean_lb[i], self['halfwidth'][i]),
									_vmap_integ_Triangular_per_dim_single(x_lb, x_ub, mean_ub[i], self['halfwidth'][i]))

			probs.append(jnp.vstack([p_min, p_max]).T)
			prob_low.append(p_min)
			prob_high.append(p_max)

		return probs, prob_low, prob_high
	
	