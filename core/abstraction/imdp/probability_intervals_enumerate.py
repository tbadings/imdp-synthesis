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
# Unlike probability_intervals.py, this version enumerates ALL partition cells for every
# state-action pair by directly using the cells stored in the partition (no max_slice window,
# no meshgrid Cartesian product). The absorbing-state probability is derived as
# 1 - sum of all per-cell probabilities.

def interval_distribution(mean_lb, mean_ub, *,
                          n, wrap, wrap_array, decimals,
                          all_cell_lb, all_cell_ub,
                          state_space_lb,
                          state_space_ub,
                          unsafe_states, noise):
    '''
    For a given state-action pair, compute the probability intervals over all successor states
    by directly enumerating the cells already stored in the partition.
    '''

    # Extract per-dimension columns from the (S, n) cell-bound arrays.
    # Each x_lb_per_dim[i] / x_ub_per_dim[i] is a 1-D array of length S (one value per cell).
    x_lb_per_dim = [all_cell_lb[:, i] for i in range(n)]
    x_ub_per_dim = [all_cell_ub[:, i] for i in range(n)]

    # Switch explicitly on noise type to avoid silently accepting unsupported distributions.
    noise_type = noise['type']
    if noise_type == 'Gaussian':
        _, prob_low_per_dim, prob_high_per_dim = noise.prob_minmax_per_dim(n, wrap, x_lb_per_dim, x_ub_per_dim, mean_lb, mean_ub, state_space_ub - state_space_lb)
        prob_state_space = noise.prob_minmax(state_space_lb, state_space_ub, mean_lb, mean_ub, wrap_array)
    elif noise_type == 'Triangular':
        _, prob_low_per_dim, prob_high_per_dim = noise.prob_minmax_per_dim(n, wrap, x_lb_per_dim, x_ub_per_dim, mean_lb, mean_ub, state_space_ub - state_space_lb)
        prob_state_space = noise.prob_minmax(state_space_lb, state_space_ub, mean_lb, mean_ub, wrap_array)
    else:
        raise ValueError(f'Unsupported noise type: {noise_type}. Expected Gaussian or Triangular.')

    # Elementwise product across dimensions — each array has shape (S,), giving one
    # probability per cell. This is correct because cells are already stored as full
    # (lb, ub) pairs; no outer product or index mapping is needed.
    prob_low_prod = prob_low_per_dim[0]
    for i in range(1, n):
        prob_low_prod = prob_low_prod * prob_low_per_dim[i]
    prob_low_prod = jnp.round(prob_low_prod, decimals)

    prob_high_prod = prob_high_per_dim[0]
    for i in range(1, n):
        prob_high_prod = prob_high_prod * prob_high_per_dim[i]
    prob_high_prod = jnp.round(prob_high_prod, decimals)

    # Cell index == region ID: the stored cells are already in the flat region ordering.
    prob_id = jnp.arange(all_cell_lb.shape[0], dtype=jnp.int32)

    p_lowest = 10 ** -decimals

    # All indices are in-bounds by construction; only filter near-zero upper bounds.
    prob_nonzero = prob_high_prod > p_lowest

    # For the nonzero probabilities, also set a (very small) minimum lower bound probability
    # (to ensure the IMDP is "graph-preserving").
    prob_low_prod = jnp.maximum(p_lowest * prob_nonzero, prob_low_prod)
    prob_high_prod = jnp.maximum(p_lowest * prob_nonzero, prob_high_prod)

    # Stack lower and upper bounds such that prob[s] is an array of length two representing a single interval.
    prob = jnp.stack([prob_low_prod, prob_high_prod]).T

    # Absorbing probability interval derived from cell probabilities:
    #   lb = max(0, 1 - sum(upper bounds))  — best case for reaching partition cells
    #   ub = max(0, 1 - sum(lower bounds))  — worst case for reaching partition cells
    # prob_absorbing_lb = jnp.maximum(0.0, jnp.round(1.0 - jnp.sum(prob_high_prod), decimals))
    # prob_absorbing_ub = jnp.maximum(0.0, jnp.round(1.0 - jnp.sum(prob_low_prod), decimals))
    # prob_absorbing = jnp.array([prob_absorbing_lb, prob_absorbing_ub])
    # prob_absorbing = jnp.maximum(p_lowest * (prob_absorbing[1] > 0), prob_absorbing)

    # TODO: This computation is currently incorrect, because not all of X might be covered by the (sparse) partition. 
    prob_absorbing = jnp.round(1 - prob_state_space[::-1], decimals)
    prob_absorbing = jnp.maximum(p_lowest * (prob_absorbing[1] > 0), prob_absorbing)

    # Keep this distribution only if the probability of reaching the absorbing state is less than given threshold.
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
    Compute probability intervals for all states and actions of the IMDP by enumerating all
    partition cells (no max_slice window).

    :param args: Argument object.
    :param model: Model object.
    :param partition: Partition object.
    :param actions: Actions object.
    :return:
        - prob: Probability intervals per state-action pair
        - prob_id: Successor states associated with these probability intervals per state-action pair
        - prob_absorbing: Probability interval of reaching the absorbing state per state-action pair
    '''

    print('Compute probability intervals for all state-action pairs (full enumeration)...')

    frs_lb = actions.frs_lb
    frs_ub = actions.frs_ub
    model_wrap_tuple = tuple(np.array(model.wrap))

    interval_distribution_fixed = partial(
        interval_distribution,
        n=model.n,
        wrap=model_wrap_tuple,
        wrap_array=model.wrap,
        decimals=args.decimals,
        all_cell_lb=jax.device_put(partition.regions['lower_bounds']),
        all_cell_ub=jax.device_put(partition.regions['upper_bounds']),
        state_space_lb=jax.device_put(partition.boundary_lb),
        state_space_ub=jax.device_put(partition.boundary_ub),
        unsafe_states=jax.device_put(partition.critical['bools']),
        noise=model.noise,
    )

    # vmap over the 2 per-action args only; all constants are captured in the closure
    vmap_interval_distribution = jax.jit(
        jax.vmap(interval_distribution_fixed, in_axes=(0, 0), out_axes=(0, 0, 0, 0, 0, 0)))

    action_labels = {}
    interval_matrix = {}
    successor_id = {}
    interval_absorbing = {}

    actions_id = np.asarray(actions.id)

    nrA = len(actions_id)
    if vectorized:

        starts, ends = create_batches(len(partition.regions['idxs']), batch_size=args.batch_size)

        for iter, (i, j) in tqdm(enumerate(zip(starts, ends)), total=len(starts)):

            # Reshape frs_lb/ub from S x A x n to (S x A) rows and n columns
            frs_lb_2D = frs_lb[i:j].reshape(-1, model.n)
            frs_ub_2D = frs_ub[i:j].reshape(-1, model.n)

            p, s_id, _, p_abs, keep_actions, number_nonzero = vmap_interval_distribution(
                                                                                    frs_lb_2D,
                                                                                    frs_ub_2D)

            # Transfer outputs from device in one call to reduce synchronization overhead.
            # jax.device_get already returns numpy arrays, so np.asarray is not needed.
            p, s_id, p_abs, keep_actions, number_nonzero = jax.device_get((p, s_id, p_abs, keep_actions, number_nonzero))
            max_nonzero = int(np.max(number_nonzero))

            # Reshape once to avoid expensive global masking/cumsum splitting.
            batch_states = j - i
            keep_actions = keep_actions.reshape(batch_states, nrA)
            p = p[:, :max_nonzero].reshape(batch_states, nrA, max_nonzero, 2)
            s_id = s_id[:, :max_nonzero].reshape(batch_states, nrA, max_nonzero)
            p_abs = np.maximum(args.pAbs_min, np.round(p_abs, args.decimals)).reshape(batch_states, nrA, 2)

            for idx, s in enumerate(range(i, j)):
                keep_mask = keep_actions[idx]
                action_labels[s] = actions_id[keep_mask]
                interval_matrix[s] = p[idx, keep_mask]
                successor_id[s] = s_id[idx, keep_mask]
                interval_absorbing[s] = p_abs[idx, keep_mask]

            del p, s_id, p_abs, keep_actions, number_nonzero

    else:

        # For all states
        for s in tqdm(range(len(partition.regions['idxs']))):

            p, s_id, _, p_abs, keep_actions, number_nonzero = vmap_interval_distribution(
                                                                                frs_lb[s].reshape(-1, model.n),
                                                                                frs_ub[s].reshape(-1, model.n))

            p, s_id, p_abs, keep_actions, number_nonzero = jax.device_get((p, s_id, p_abs, keep_actions, number_nonzero))
            max_nonzero = int(np.max(number_nonzero))

            # k=True are the action indices that are to be kept (i.e., those with nonzero probabilities and for which the absorbing state probability is less than threshold)
            # p_nonzero=True means that the upper bound of the probability interval is greater than the minimum probability threshold
            # Evaluate p_nonzero over each columns to get the successor states that we should keep
            if np.any(keep_actions):
                action_labels[s] = actions_id[keep_actions]
                # Trim the trailing dimension before fancy-indexing to avoid a large intermediate copy.
                interval_matrix[s] = p[:, :max_nonzero][keep_actions]
                successor_id[s] = s_id[:, :max_nonzero][keep_actions]
                interval_absorbing[s] = np.maximum(args.pAbs_min, np.round(p_abs[keep_actions], args.decimals))
            del p, s_id, p_abs

    print('-- Number of times function was compiled:', vmap_interval_distribution._cache_size())
    print('')

    return interval_matrix, successor_id, action_labels, interval_absorbing
