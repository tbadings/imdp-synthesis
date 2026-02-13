from functools import partial, reduce

from core.utils import create_batches
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from collections import ChainMap
from itertools import chain
import time
import sys

# Note: The following implementation only works for Gaussian distributions with diagonal covariance

def dynslice(V, idx_low, size):
    '''
    Given a vector of indices, keep only those starting at position idx_low and of length size.

    :param V: Vector of indices.
    :param idx_low: Index to start slice at.
    :param size: Number of elements to keep in slice.
    :return: Slice of V.
    '''
    roll = jnp.roll(V, -idx_low)
    # roll_zero = roll.at[size:].set(0)
    return roll[:size]

def integ_Gauss(x_lb, x_ub, x, cov):
    '''
    Integrate a univariate Gaussian distribution over a given interval.

    :param x_lb: Lower bound of the interval.
    :param x_ub: Upper bound of the interval.
    :param x: Mean of the Gaussian distribution.
    :param cov: Covariance of the Gaussian distribution.
    :return:
    '''
    eps = 1e-4  # Add tiny epsilon to avoid NaN problems if the Gaussian is a Dirac (i.e., cov=0) and x_lb or x_ub equals x
    return jax.scipy.stats.norm.cdf(x_ub, x + eps, cov) - jax.scipy.stats.norm.cdf(x_lb, x + eps, cov)


# vmap to compute multivariate Gaussian integral in n dimensions
vmap_integ_Gauss = jax.jit(jax.vmap(integ_Gauss, in_axes=(0, 0, 0, 0), out_axes=0))
vmap_integ_Gauss_per_dim = jax.jit(jax.vmap(integ_Gauss, in_axes=(0, 0, 0, None), out_axes=0))
vmap_integ_Gauss_per_dim_single = jax.jit(jax.vmap(integ_Gauss, in_axes=(0, 0, None, None), out_axes=0))

def minmax_Gauss(x_lb, x_ub, mean_lb, mean_ub, cov, wrap_array):
    '''
    Compute the min/max integral of a multivariate Gaussian distribution over a given interval, where the mean of the Gaussian lies in [mean_lb, mean_ub].

    :param x_lb: Lower bound of the interval.
    :param x_ub: Upper bound of the interval.
    :param mean_lb: Lower bound of the mean of the Gaussian distribution.
    :param mean_ub: Upper bound of the mean of the Gaussian distribution.
    :param cov: Covariance of the Gaussian distribution.
    :param wrap_array: Wrap at the indices where this array is True.
    :return: Min/max probabilities.
    '''

    # Determine point closest to mean of region over which to integrate
    mean = (x_lb + x_ub) / 2
    closest_to_mean = jnp.maximum(jnp.minimum(mean_ub, mean), mean_lb)

    # Maximum probability is the product
    p_max = jnp.prod(vmap_integ_Gauss(x_lb, x_ub, closest_to_mean, jnp.diag(cov)) * ~wrap_array + 1 * wrap_array)

    p1 = vmap_integ_Gauss(x_lb, x_ub, mean_lb, jnp.diag(cov)) * ~wrap_array + 1 * wrap_array
    p2 = vmap_integ_Gauss(x_lb, x_ub, mean_ub, jnp.diag(cov)) * ~wrap_array + 1 * wrap_array
    p_min = jnp.prod(jnp.minimum(p1, p2))

    return jnp.array([p_min, p_max])

def minmax_Gauss_per_dim(n, wrap, x_lb_per_dim, x_ub_per_dim, mean_lb, mean_ub, cov, state_space_size):
    '''
    Compute the min/max integral of a multivariate Gaussian distribution over a given interval, where the mean of the Gaussian lies in [mean_lb, mean_ub].
    Exploit rectangular partition to compute much fewer Gaussian integrals

    :param n: Dimension of the state space.
    :param wrap: Wrap at the indices where this array is True.
    :param x_lb_per_dim: Lower bound of the interval per dimension.
    :param x_ub_per_dim: Upper bound of the interval per dimension.
    :param mean_lb: Lower bound of the mean of the Gaussian distribution.
    :param mean_ub: Upper bound of the mean of the Gaussian distribution.
    :param cov: Covariance of the Gaussian distribution.
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
        mean = (x_lb + x_ub) / 2
        closest_to_mean = jnp.maximum(jnp.minimum(mean_ub[i], mean), mean_lb[i])

        if wrap[i]:
            p_max = sum(vmap_integ_Gauss_per_dim(x_lb + shift, x_ub + shift, closest_to_mean, cov[i, i]) 
                         for shift in [-state_space_size[i], 0, state_space_size[i]])
            p_min = sum(jnp.minimum(vmap_integ_Gauss_per_dim_single(x_lb + shift, x_ub + shift, mean_lb[i], cov[i, i]),
                                    vmap_integ_Gauss_per_dim_single(x_lb + shift, x_ub + shift, mean_ub[i], cov[i, i]))
                         for shift in [-state_space_size[i], 0, state_space_size[i]])
        else:
            p_max = vmap_integ_Gauss_per_dim(x_lb, x_ub, closest_to_mean, cov[i, i])
            p_min = jnp.minimum(vmap_integ_Gauss_per_dim_single(x_lb, x_ub, mean_lb[i], cov[i, i]),
                                vmap_integ_Gauss_per_dim_single(x_lb, x_ub, mean_ub[i], cov[i, i]))

        probs.append(jnp.vstack([p_min, p_max]).T)
        prob_low.append(p_min)
        prob_high.append(p_max)

    return probs, prob_low, prob_high


@partial(jax.jit, static_argnums=(0, 1, 2, 4))
def interval_distribution_per_dim(n, max_slice, wrap, wrap_array, decimals, number_per_dim, per_dim_lb, per_dim_ub, i_lb, mean_lb, mean_ub, cov, state_space_lb, state_space_ub,
                                  region_idx_array, unsafe_states):
    '''
    For a given state-action pair, compute the probability intervals over all successor states.
    '''

    # Extract slices from the partition elements per dimension
    x_lb = [dynslice(per_dim_lb[i], i_lb[i], max_slice[i]) for i in range(n)]
    x_ub = [dynslice(per_dim_ub[i], i_lb[i], max_slice[i]) for i in range(n)]

    # List of indexes of the partition elements in the slices above
    prob_idx = [jnp.arange(max_slice[i]) + i_lb[i] for i in range(n)]

    # Compute the probability intervals for each dimension
    _, prob_low, prob_high = minmax_Gauss_per_dim(n, wrap, x_lb, x_ub, mean_lb, mean_ub, cov, state_space_ub - state_space_lb)

    prob_low_prod = jnp.round(reduce(jnp.multiply.outer, prob_low).flatten(), decimals)
    prob_high_prod = jnp.round(reduce(jnp.multiply.outer, prob_high).flatten(), decimals)

    # Note: meshgrid is used to get the Cartesian product between the indexes of the partition elements in every state space dimension, but meshgrid sorts in the wrong order.
    # To fix this, we first flip the order of the dimensions, then compute the meshgrid, and again flip the columns of the result. This ensures the sorting is in the correct order.
    prob_idx_flip = [prob_idx[n - i - 1] for i in range(n)]
    prob_idx = jnp.flip(jnp.asarray(jnp.meshgrid(*prob_idx_flip, indexing='ij')).T.reshape(-1, n), axis=1)

    prob_idx_clip = jnp.clip(prob_idx, 0, number_per_dim).astype(int)
    prob_id = region_idx_array[tuple(prob_idx_clip.T)]

    p_lowest = 10 ** -decimals
    # Only keep nonzero probabilities
    prob_nonzero = (prob_high_prod > p_lowest) & jnp.all(prob_idx < number_per_dim, axis=1)

    # Set a minimum lower bound probability
    prob_low_prod = jnp.where(prob_nonzero, jnp.maximum(p_lowest, prob_low_prod), prob_low_prod)
    prob_high_prod = jnp.where(prob_nonzero, jnp.maximum(p_lowest, prob_high_prod), prob_high_prod)

    # Stack lower and upper bounds
    prob = jnp.stack([prob_low_prod, prob_high_prod]).T

    # Compute probability to end outside of partition
    prob_state_space = minmax_Gauss(state_space_lb, state_space_ub, mean_lb, mean_ub, cov, wrap_array)
    prob_absorbing = jnp.round(1 - prob_state_space[::-1], decimals)
    prob_absorbing = jnp.maximum(p_lowest * (prob_absorbing[1] > 0), prob_absorbing)

    # Keep this distribution only if the probability of reaching the absorbing state is less than given threshold
    threshold = 0.1
    unsafe_states_slice = unsafe_states[prob_id]
    keep = ~(((jnp.sum(prob[:, 0] * ~unsafe_states_slice)) < 1 - threshold) & ((prob_absorbing[1] + jnp.sum(prob[:, 1] * unsafe_states_slice)) > threshold))

    number_nonzero = jnp.sum(prob_nonzero)
    
    # TODO: argsort here is slow...
    sorted_idx = jnp.argsort(prob_nonzero, axis=0)[::-1]
    prob = prob[sorted_idx]
    prob_id = prob_id[sorted_idx]
    prob_nonzero = prob_nonzero[sorted_idx]
    
    return prob, prob_id, prob_nonzero, prob_absorbing, keep, number_nonzero

def compute_probability_intervals(args, model, partition, actions):
    '''
    Compute probability intervals for all states and actions of the IMDP.

    :param args: Argument object.
    :param model: Model object.
    :param partition: Partition object.
    :param frs: Forward reachable sets.
    :param max_slice: Array where each element is the maximum number of partition elements to consider in each dimension.
    :return:
        - prob: Probability intervals per state-action pair
        - prob_id: Successor states associated with these probability intervals per state-action pair
        - prob_absorbing: Probability interval of reaching the absorbing state per state-action pair
    '''

    frs_lb = actions.frs_lb
    frs_ub = actions.frs_ub
    frs_idx_lb = actions.frs_idx_lb

    # vmap to compute distributions for all actions in a state
    vmap_interval_distribution_per_dim = jax.jit(
        jax.vmap(interval_distribution_per_dim, in_axes=(None, None, None, None, None, None, None, None, 0, 0, 0, None, None, None, None, None), out_axes=(0, 0, 0, 0, 0, 0)),
        static_argnums=(0, 1, 2, 4))

    pAbs_min = 0.0001

    starts, ends = create_batches(len(partition.regions['idxs']), batch_size=args.batch_size)
    
    list_prob = [[]]*len(starts)
    list_prob_absorbing = [[]]*len(starts)
    list_prob_id = [[]]*len(starts)

    for iter, (i, j) in enumerate(zip(starts, ends)):
        print('- Compute probability intervals for states {} to {}... (out of {})'.format(i, j - 1, len(partition.regions['idxs'])))
        
        keep = {}
        prob = {}
        prob_id = {}
        prob_nonzero = {}
        prob_absorbing = {}

        # For all states
        for s in tqdm(range(i,j), total=j-i):
            p, p_id, p_nonzero, pa, k, _ = vmap_interval_distribution_per_dim(model.n,
                                                                                actions.max_slice,
                                                                                tuple(np.array(model.wrap)),
                                                                                model.wrap,
                                                                                args.decimals,
                                                                                partition.number_per_dim,
                                                                                partition.regions_per_dim['lower_bounds'],
                                                                                partition.regions_per_dim['upper_bounds'],
                                                                                frs_idx_lb[s],
                                                                                frs_lb[s],
                                                                                frs_ub[s],
                                                                                model.noise['cov'],
                                                                                partition.boundary_lb,
                                                                                partition.boundary_ub,
                                                                                partition.region_idx_array,
                                                                                partition.critical['bools'])
                                                                                
            # print(p.shape, p_idx.shape, p_id.shape, p_nonzero.shape, pa.shape, k.shape)

            keep[s] = np.array(k, dtype=bool)
            prob[s] = np.array(p)
            prob_id[s] = np.array(p_id)
            prob_nonzero[s] = np.array(p_nonzero)
            prob_absorbing[s] = np.round(np.array(pa), args.decimals)

            nans = np.where(np.any(np.isnan(prob[s]), axis=0))[0]
            if len(nans) > 0:
                print('NaN probabilities in state {} at position {}'.format(s, len(nans)))

        list_prob[iter] = [[np.round(val[prob_nonzero[s][a]], args.decimals) for a, val in enumerate(row) if keep[s][a]] for s, row in prob.items()]
        list_prob_absorbing[iter] = [[np.maximum(pAbs_min, np.round(val, args.decimals)) for a, val in enumerate(row) if keep[s][a]] for s, row in prob_absorbing.items()]
        list_prob_id[iter] = {s: {a: val[prob_nonzero[s][a]] for a, val in enumerate(row) if keep[s][a]} for s, row in prob_id.items()}

    # Merge list_prob over batches
    prob = list(chain(*list_prob))
    prob_absorbing = list(chain(*list_prob_absorbing))
    prob_id = dict(ChainMap(*list_prob_id))

    print('-- Number of times function was compiled:', vmap_interval_distribution_per_dim._cache_size())

    return prob, prob_id, prob_absorbing

def compute_probability_intervals_vec(args, model, partition, actions, batch_size=1000):
    '''
    Compute probability intervals for all states and actions of the IMDP.

    :param args: Argument object.
    :param model: Model object.
    :param partition: Partition object.
    :param frs: Forward reachable sets.
    :param max_slice: Array where each element is the maximum number of partition elements to consider in each dimension.
    :return:
        - prob: Probability intervals per state-action pair
        - prob_id: Successor states associated with these probability intervals per state-action pair
        - prob_absorbing: Probability interval of reaching the absorbing state per state-action pair
    '''

    frs_lb = actions.frs_lb
    frs_ub = actions.frs_ub
    frs_idx_lb = actions.frs_idx_lb

    vmap_interval_distribution_per_dim = jax.jit(
            jax.vmap(interval_distribution_per_dim, in_axes=(None, None, None, None, None, None, None, None, 0, 0, 0, None, None, None, None, None), out_axes=(0, 0, 0, 0, 0, 0)),
            static_argnums=(0, 1, 2, 4))

    pAbs_min = 0.0001

    max_successors = np.prod(np.array(actions.max_slice))

    state_indexes_ij = np.array([(s, a) for s in range(len(partition.regions['idxs'])) for a in range(len(actions.inputs))])
    # Create matrix of same number of rows, with 1 in first column and 0 in the second
    state_indexes_offset = np.hstack([np.ones((state_indexes_ij.shape[0], 1), dtype=int), np.zeros((state_indexes_ij.shape[0], 1), dtype=int)])

    # Reshape frs_idx_lb from S x A x n to (S x A) rows and n columns
    frs_idx_lb_2D = frs_idx_lb.reshape(-1, model.n)
    frs_lb_2D = frs_lb.reshape(-1, model.n)
    frs_ub_2D = frs_ub.reshape(-1, model.n)

    starts, ends = create_batches(len(frs_idx_lb_2D), batch_size=batch_size)

    for iter, (i, j) in tqdm(enumerate(zip(starts, ends)), total=len(starts)):

        p, p_id, p_nonzero, pa, k, number_nonzero = vmap_interval_distribution_per_dim(model.n,
                                                                    actions.max_slice,
                                                                    tuple(np.array(model.wrap)),
                                                                    model.wrap,
                                                                    args.decimals,
                                                                    partition.number_per_dim,
                                                                    partition.regions_per_dim['lower_bounds'],
                                                                    partition.regions_per_dim['upper_bounds'],
                                                                    frs_idx_lb_2D[i:j],
                                                                    frs_lb_2D[i:j],
                                                                    frs_ub_2D[i:j],
                                                                    model.noise['cov'],
                                                                    partition.boundary_lb,
                                                                    partition.boundary_ub,
                                                                    partition.region_idx_array,
                                                                    partition.critical['bools'])
        jnp.array([1])

        max_number_nonzero = jnp.max(number_nonzero)
        p = p[k, :max_number_nonzero, :]
        p_id = p_id[k, :max_number_nonzero]
        p_nonzero = p_nonzero[k, :max_number_nonzero]
        pa = pa[k]
        state_action_indexes = state_indexes_ij[i:j][k]

    print('-- Number of times function was compiled:', vmap_interval_distribution_per_dim._cache_size())

    return 0,0,0