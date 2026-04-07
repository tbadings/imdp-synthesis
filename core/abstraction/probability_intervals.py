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

# Note: The following implementation supports Gaussian and Triangular noise distributions.

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

def interval_distribution_per_dim(n, max_slice, wrap, wrap_array, decimals, number_per_dim, per_dim_lb, per_dim_ub, i_lb, mean_lb, mean_ub, state_space_lb, state_space_ub,
                                  region_idx_array, unsafe_states, noise):
    '''
    For a given state-action pair, compute the probability intervals over all successor states.
    '''

    # Extract slices from the partition elements per dimension
    x_lb = [dynslice(per_dim_lb[i], i_lb[i], max_slice[i]) for i in range(n)]
    x_ub = [dynslice(per_dim_ub[i], i_lb[i], max_slice[i]) for i in range(n)]

    # List of indexes of the partition elements in the slices above
    prob_idx = [jnp.arange(max_slice[i]) + i_lb[i] for i in range(n)]

    # Switch explicitly on noise type to avoid silently accepting unsupported distributions.
    noise_type = noise['type']
    if noise_type == 'Gaussian':
        _, prob_low, prob_high = noise.prob_minmax_per_dim(n, wrap, x_lb, x_ub, mean_lb, mean_ub, state_space_ub - state_space_lb)
        prob_state_space = noise.prob_minmax(state_space_lb, state_space_ub, mean_lb, mean_ub, wrap_array)
    elif noise_type == 'Triangular':
        _, prob_low, prob_high = noise.prob_minmax_per_dim(n, wrap, x_lb, x_ub, mean_lb, mean_ub, state_space_ub - state_space_lb)
        prob_state_space = noise.prob_minmax(state_space_lb, state_space_ub, mean_lb, mean_ub, wrap_array)
    else:
        raise ValueError(f'Unsupported noise type: {noise_type}. Expected Gaussian or Triangular.')

    prob_low_prod = prob_low[0]
    for i in range(1, n):
        prob_low_prod = prob_low_prod[..., None] * prob_low[i]
    prob_low_prod = jnp.round(prob_low_prod.reshape(-1), decimals)

    prob_high_prod = prob_high[0]
    for i in range(1, n):
        prob_high_prod = prob_high_prod[..., None] * prob_high[i]
    prob_high_prod = jnp.round(prob_high_prod.reshape(-1), decimals)

    # Note: meshgrid is used to get the Cartesian product between the indexes of the partition elements in every state space dimension, but meshgrid sorts in the wrong order.
    # To fix this, we first flip the order of the dimensions, then compute the meshgrid, and again flip the columns of the result. This ensures the sorting is in the correct order.
    prob_idx_flip = [prob_idx[n - i - 1] for i in range(n)]
    prob_idx = jnp.flip(jnp.asarray(jnp.meshgrid(*prob_idx_flip, indexing='ij')).T.reshape(-1, n), axis=1)

    prob_idx_clip = jnp.astype(jnp.clip(prob_idx, jnp.zeros(n), number_per_dim), int)
    prob_id = region_idx_array[tuple(prob_idx_clip.T)]

    p_lowest = 10 ** -decimals
    
    # Only keep nonzero probabilities, and also filter spurious indices that were added to keep arrays in JAX of fixed size
    prob_nonzero = (prob_high_prod > p_lowest) * jnp.all(prob_idx < number_per_dim, axis=1)

    # For the nonzero probabilities, also set a (very small) minimum lower bound probability (to ensure the IMDP is "graph-preserving")
    prob_low_prod = jnp.maximum(p_lowest * prob_nonzero, prob_low_prod)
    prob_high_prod = jnp.maximum(p_lowest * prob_nonzero, prob_high_prod)

    # Stack lower and upper bounds such that such prob[s] is an array of length two representing a single interval
    prob = jnp.stack([prob_low_prod, prob_high_prod]).T

    # Compute probability to end outside of partition
    prob_absorbing = jnp.round(1 - prob_state_space[::-1], decimals)
    prob_absorbing = jnp.maximum(p_lowest * (prob_absorbing[1] > 0), prob_absorbing)

    # Keep this distribution only if the probability of reaching the absorbing state is less than given threshold
    threshold = 0.1
    unsafe_states_slice = unsafe_states[prob_id]
    keep = ~(((jnp.sum(prob[:, 0] * ~unsafe_states_slice)) < 1 - threshold) * ((prob_absorbing[1] + jnp.sum(prob[:, 1] * unsafe_states_slice)) > threshold))

    number_nonzero = jnp.sum(prob_nonzero)
    
    # Move all nonzero probabilities to the front without a full sort.
    # TODO: Fix bug that causes wrong IDs that are supposed to be excluded (but they don't hurt, because their probabilities are zero anyway).
    n_probs = prob_nonzero.shape[0]
    idx = jnp.arange(n_probs, dtype=jnp.int32)
    true_pos = jnp.cumsum(prob_nonzero) - 1
    false_pos = jnp.cumsum(~prob_nonzero) - 1
    num_true = jnp.sum(prob_nonzero)
    target_pos = jnp.where(prob_nonzero, true_pos, num_true + false_pos)
    sorted_idx = jnp.empty_like(idx)
    sorted_idx = sorted_idx.at[target_pos].set(idx)
    prob = prob[sorted_idx]
    prob_id = prob_id[sorted_idx]
    prob_nonzero = prob_nonzero[sorted_idx]
    
    return prob, prob_id, prob_nonzero, prob_absorbing, keep, number_nonzero

def compute_probability_intervals(args, model, partition, actions, vectorized=True):
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

    print('Compute probability intervals for all state-action pairs...')

    frs_lb = actions.frs_lb
    frs_ub = actions.frs_ub
    frs_idx_lb = actions.frs_idx_lb
    model_wrap_tuple = tuple(np.array(model.wrap))

    interval_distribution_per_dim_noise = partial(interval_distribution_per_dim, noise=model.noise)

    # vmap to compute distributions for all actions in a state
    vmap_interval_distribution_per_dim = jax.jit(
        jax.vmap(interval_distribution_per_dim_noise, in_axes=(None, None, None, None, None, None, None, None, 0, 0, 0, None, None, None, None), out_axes=(0, 0, 0, 0, 0, 0)),
        static_argnums=(0, 1, 2, 4))

    action_labels = {}
    interval_matrix = {}
    successor_id = {}
    interval_absorbing = {}

    # Repeat actions.id batch_size number of times
    a_id_repeated = np.tile(actions.id, args.batch_size)

    JAX_boundary_lb = jax.device_put(partition.boundary_lb)
    JAX_boundary_ub = jax.device_put(partition.boundary_ub)
    JAX_region_idx_array = jax.device_put(partition.region_idx_array)
    JAX_unsafe_states = jax.device_put(partition.critical['bools'])

    nrA = len(actions.id)
    if vectorized:

        starts, ends = create_batches(len(partition.regions['idxs']), batch_size=args.batch_size)

        for iter, (i, j) in tqdm(enumerate(zip(starts, ends)), total=len(starts)):

            # Reshape frs_idx_lb from S x A x n to (S x A) rows and n columns
            frs_idx_lb_2D = frs_idx_lb[i:j].reshape(-1, model.n)
            frs_lb_2D = frs_lb[i:j].reshape(-1, model.n)
            frs_ub_2D = frs_ub[i:j].reshape(-1, model.n)

            p, s_id, _, p_abs, keep_actions, number_nonzero = vmap_interval_distribution_per_dim(model.n,
                                                                                    actions.max_slice,
                                                                                    model_wrap_tuple,
                                                                                    model.wrap,
                                                                                    args.decimals,
                                                                                    partition.number_per_dim,
                                                                                    partition.regions_per_dim['lower_bounds'],
                                                                                    partition.regions_per_dim['upper_bounds'],
                                                                                    frs_idx_lb_2D, # Vectorized over multiple states
                                                                                    frs_lb_2D, # Vectorized over multiple states
                                                                                    frs_ub_2D, # Vectorized over multiple states
                                                                                    JAX_boundary_lb,
                                                                                    JAX_boundary_ub,
                                                                                    JAX_region_idx_array,
                                                                                    JAX_unsafe_states)

            # Transfer outputs from device in one call to reduce synchronization overhead.
            # jax.device_get already returns numpy arrays, so np.asarray is not needed.
            p, s_id, p_abs, keep_actions, number_nonzero = jax.device_get((p, s_id, p_abs, keep_actions, number_nonzero))
            max_nonzero = int(np.max(number_nonzero))

            # If not final iteration
            if iter < len(starts) - 1:
                batch_action_labels = a_id_repeated[keep_actions]
            else:
                batch_action_labels = np.tile(actions.id, j-i)[keep_actions]

            # Trim the trailing dimension before fancy-indexing to avoid a large intermediate copy.
            # p[:, :max_nonzero] is a free view; the copy is only (kept, max_nonzero, 2) instead of (kept, total_cells, 2).
            batch_interval_matrix = p[:, :max_nonzero][keep_actions]
            del p
            batch_successor_id = s_id[:, :max_nonzero][keep_actions]
            del s_id
            batch_interval_absorbing = np.maximum(args.pAbs_min, np.round(p_abs[keep_actions], args.decimals))
            del p_abs

            # Take cumsum of actions to know where to split the batch results for each state
            keep_actions_cumsum = np.cumsum(keep_actions)

            for idx,s in enumerate(range(i,j)):
                # Number of actions to keep for this state
                start = keep_actions_cumsum[nrA * idx - 1] if idx > 0 else 0
                end = keep_actions_cumsum[nrA * (idx + 1) - 1]

                action_labels[s] = batch_action_labels[start:end]
                interval_matrix[s] = batch_interval_matrix[start:end]
                successor_id[s] = batch_successor_id[start:end]
                interval_absorbing[s] = batch_interval_absorbing[start:end]

    else:

        #####

        # For all states
        for s in tqdm(range(len(partition.regions['idxs']))):

            p, s_id, _, p_abs, keep_actions, number_nonzero = vmap_interval_distribution_per_dim(model.n,
                                                                                actions.max_slice,
                                                                                model_wrap_tuple,
                                                                                model.wrap,
                                                                                args.decimals,
                                                                                partition.number_per_dim,
                                                                                partition.regions_per_dim['lower_bounds'],
                                                                                partition.regions_per_dim['upper_bounds'],
                                                                                frs_idx_lb[s].reshape(-1, model.n),
                                                                                frs_lb[s].reshape(-1, model.n),
                                                                                frs_ub[s].reshape(-1, model.n),
                                                                                JAX_boundary_lb,
                                                                                JAX_boundary_ub,
                                                                                JAX_region_idx_array,
                                                                                JAX_unsafe_states)

            p, s_id, p_abs, keep_actions, number_nonzero = jax.device_get((p, s_id, p_abs, keep_actions, number_nonzero))
            max_nonzero = int(np.max(number_nonzero))
            
            # k=True are the action indices that are to be kept (i.e., those with nonzero probabilities and for which the absorbing state probability is less than threshold)
            # p_nonzero=True means that the upper bound of the probability interval is greater than the minimum probability threshold
            # Evaluate p_nonzero over each columns to get the successor states that we should keep
            if np.any(keep_actions):
                action_labels[s] = actions.id[keep_actions]
                # Trim the trailing dimension before fancy-indexing to avoid a large intermediate copy.
                interval_matrix[s] = p[:, :max_nonzero][keep_actions]
                successor_id[s] = s_id[:, :max_nonzero][keep_actions]
                interval_absorbing[s] = np.maximum(args.pAbs_min, np.round(p_abs[keep_actions], args.decimals))
            del p, s_id, p_abs

    print('-- Number of times function was compiled:', vmap_interval_distribution_per_dim._cache_size())
    print('')

    return interval_matrix, successor_id, action_labels, interval_absorbing