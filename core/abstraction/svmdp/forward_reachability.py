import logging
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from core.utils import create_batches

logger = logging.getLogger(__name__)

# Largest grid dimension that fits int8 grid-index storage. Indices are stored unclipped and may be
# negative (out-of-grid successors map to the absorbing state downstream), so the sign must be preserved.
INT8_MAX = int(np.iinfo(np.int8).max)


def _successor_index_bounds(state_min, state_max, input, step_set, cell_width, boundary_lb, shrink_frs,
                            noise_lb, noise_ub):
    """
    Grid-index interval of the successors in every dimension, for each noise interval of that dimension.

    The (noise-free) forward reachable set of the state box under the input is shifted by the noise interval
    and mapped to the grid cells it overlaps. Indices are left unclipped: they may fall outside
    [0, number_per_dim - 1], which is resolved downstream (out-of-grid successors map to the absorbing state,
    wrapped dims are taken modulo).

    :return: Per dimension d a tuple (lb, ub) of int arrays of shape [n_d] (n_d noise intervals in d). As the
        noise intervals are ascending, lb and ub are non-decreasing.
    """
    # Continuous bounds of the (noise-free) forward reachable set, shrunk slightly for numerical
    # stability (avoids issues when the FRS lands exactly on a cell boundary). epsilon is currently 0.
    epsilon = 0.0
    frs_min, frs_max = step_set(state_min, state_max, input - epsilon, input + epsilon)
    frs_min = frs_min + shrink_frs
    frs_max = frs_max - shrink_frs

    return [(jnp.floor((frs_min[d] + noise_lb[d] - boundary_lb[d]) / cell_width[d]).astype(int),
             jnp.floor((frs_max[d] + noise_ub[d] - boundary_lb[d]) / cell_width[d]).astype(int))
            for d in range(len(noise_lb))]


def count_successor_intervals(state_min, state_max, input, step_set, cell_width, boundary_lb, shrink_frs,
                              noise_lb, noise_ub):
    """
    Number of merged successor intervals, their largest span, and the span of their union, per dimension
    (as successor_intervals computes them, but without merging). Used as a cheap first pass to size the
    stored arrays, so it must use the same arithmetic.

    The arguments are those of _successor_index_bounds.

    :return: Tuple of three int arrays of shape [state_dim]: (number of merged intervals, largest interval
        span, span of the union of the intervals)
    """
    num, span, union = [], [], []
    for lb, ub in _successor_index_bounds(state_min, state_max, input, step_set, cell_width, boundary_lb,
                                          shrink_frs, noise_lb, noise_ub):
        # The merged intervals are the runs of equal (lb, ub) pairs, so they have the same spans; lb and ub
        # are non-decreasing, so the union runs from the first lb to the last ub.
        num.append(1 + jnp.sum((lb[1:] != lb[:-1]) | (ub[1:] != ub[:-1]))) # Number of different (lb, ub) pairs after merging
        span.append(jnp.max(ub - lb + 1)) # Largest span of a single interval
        union.append(ub[-1] - lb[0] + 1) # Span of the union of all intervals
    return jnp.stack(num), jnp.stack(span), jnp.stack(union)


def successor_intervals(state_min, state_max, input, step_set, cell_width, boundary_lb, shrink_frs,
                        noise_lb, noise_ub, noise_probs, slots):
    """
    Successor intervals of a state box under a control input, merged per dimension, and the probability of
    every successor box they compose.

    A noise cell's successor box is the product of one grid-index interval per dimension, and each interval
    depends only on the cell's noise interval in that dimension. Noise intervals of a dimension that give the
    same grid-index interval are merged, so a state-action pair's successor boxes are all combinations of one
    merged interval per dimension. They are not composed here; the DP takes the minimum over each box from
    the per-dimension intervals.

    The first three arguments (state_min, state_max, input) are the only ones that vary across the state
    regions / actions loop; the remaining ones are constant and bound once via functools.partial.

    :param state_min: Lower bound of the state box to propagate (shape: [state_dim])
    :param state_max: Upper bound of the state box to propagate (shape: [state_dim])
    :param input: Control input for the dynamical system (shape: [input_dim])
    :param step_set: Function (state_min, state_max, input_min, input_max) -> (next_min, next_max)
    :param cell_width: Width of grid cells along each dimension (shape: [state_dim])
    :param boundary_lb: Lower bound of the state space grid (shape: [state_dim])
    :param shrink_frs: Amount to shrink the forward reachable set by on each side (scalar)
    :param noise_lb: Per-dimension lower bounds of the noise intervals, in ascending order (tuple of
        state_dim arrays, shape [n_d] each). The noise cells are all combinations of one interval per dimension.
    :param noise_ub: Per-dimension upper bounds of the noise intervals (tuple of state_dim arrays, shape [n_d] each)
    :param noise_probs: Per-dimension probability mass of the noise intervals; a noise cell's probability is
        the product over dimensions (tuple of state_dim arrays, shape [n_d] each)
    :param slots: Static number of stored intervals per dimension (tuple of state_dim ints), at least the
        number of merged intervals of every state-action pair (see RectangularForward)
    :return: Tuple (lb, ub, probs):
        - lb, ub: Grid-index bounds of the merged intervals, dimension after dimension (shape: [sum(slots)]).
          Per dimension they are ascending and packed at the top; the unused slots repeat the last merged
          interval, so every slot is a valid interval and the union of the slots is the union of the merged
          intervals.
        - probs: Probability of every successor box, i.e. of every combination of one slot per dimension, in
          C order over the dimensions (shape: [prod(slots)]). Boxes with an unused slot have probability 0.
    """
    lbs, ubs, merged_probs = [], [], []
    for d, (lb, ub) in enumerate(_successor_index_bounds(state_min, state_max, input, step_set, cell_width,
                                                         boundary_lb, shrink_frs, noise_lb, noise_ub)):
        if lb.shape[0] == 1:
            # A single noise interval (e.g. a dimension without noise) is always one merged interval.
            lbs.append(lb)
            ubs.append(ub)
            merged_probs.append(noise_probs[d])
            continue

        # Equal (lb, ub) pairs are adjacent: label the start of each run, pack the runs at the top, sum their
        # probabilities, and fill the unused slots with the last run (probability 0).
        is_first = jnp.concatenate([jnp.array([True]), (lb[1:] != lb[:-1]) | (ub[1:] != ub[:-1])])
        run = jnp.cumsum(is_first) - 1
        last = jnp.minimum(jnp.arange(slots[d]), run[-1])
        lbs.append(jnp.zeros_like(lb).at[run].set(lb)[last])
        ubs.append(jnp.zeros_like(ub).at[run].set(ub)[last])
        merged_probs.append(jnp.zeros(slots[d], noise_probs[d].dtype).at[run].add(noise_probs[d]))

    # Probability of each successor box. The noise dimensions are independent, so it is the product of its
    # intervals' probabilities. (For correlated noise, it would instead be the joint noise probability summed
    # over the noise cells of the merged runs.)
    probs = merged_probs[0]
    for p in merged_probs[1:]:
        probs = probs[..., None] * p

    return jnp.concatenate(lbs), jnp.concatenate(ubs), probs.reshape(-1)


class RectangularForward(object):
    """
    Computes and stores forward reachable sets for a rectangular partition of the state space.

    This class pre-computes the successors of all state regions in a partition under all discrete control
    actions. Per (state, action) and dimension it stores the merged successor intervals (one per distinct
    grid-index interval over the noise intervals of that dimension); the successor boxes are all combinations
    of one interval per dimension, and their probabilities are stored per box.

    Attributes:
        interval_lb (np.ndarray): Lower grid indices of the merged successor intervals, dimension after
            dimension, shape [num_regions, num_actions, sum(slots)], dtype int8 or int16
        interval_ub (np.ndarray): Upper grid indices of the merged successor intervals, same shape and dtype
        box_probs (np.ndarray): Probability of each successor box (combination of one interval per dimension,
            C order over the dimensions), shape [num_regions, num_actions, prod(slots)]
        slots (tuple): Number of stored intervals per dimension (the most merged intervals of any pair)
        max_slice (tuple): Largest span of a successor interval per dimension
        union_span (tuple): Largest span of the union of a pair's successor intervals per dimension
        id (np.ndarray): Indices of all actions, shape [num_actions]
    """

    def __init__(self, args, partition, model):
        """
        Initialize and compute forward reachable sets for all regions and actions.

        :param args: Argument namespace (provides shrink_frs, frs_batch_size, floatprecision)
        :param partition: Partition object containing the discretized state space
        :param model: Model object containing the dynamics and control action specifications
        """
        logger.info('=== Start forward reachability computations ===')
        t_total = time.time()

        # Noise partition: a grid of per-dimension intervals with product probabilities.
        per_dim_cells = [np.asarray(c) for c in model.noise.partition['per_dim_cells']]  # D x (n_d, 2)
        per_dim_probs = [np.asarray(p) for p in model.noise.partition['per_dim_probs']]  # D x (n_d,)
        # The per-dimension merge only compares neighbouring intervals, so they must be ascending.
        assert all(np.all(np.diff(c[:, 0]) > 0) for c in per_dim_cells)
        noise_lb_dev = tuple(jax.device_put(c[:, 0]) for c in per_dim_cells)
        noise_ub_dev = tuple(jax.device_put(c[:, 1]) for c in per_dim_cells)
        noise_probs_dev = tuple(jax.device_put(p) for p in per_dim_probs)

        # Constant arguments, bound once; only (state_min, state_max, input) vary in the loop.
        bound = dict(step_set=model.step_set, cell_width=jax.device_put(partition.cell_width),
                     boundary_lb=jax.device_put(partition.boundary_lb), shrink_frs=args.shrink_frs,
                     noise_lb=noise_lb_dev, noise_ub=noise_ub_dev)

        # Inner vmap over control actions, outer vmap over a batch of state regions; only the three
        # varying arguments are mapped. This reduces Python–JAX round trips from num_regions to
        # ceil(num_regions / frs_batch_size).
        def over_states_and_actions(fn):
            return jax.vmap(jax.vmap(fn, in_axes=(None, None, 0)), in_axes=(0, 0, 0))

        count_fn = over_states_and_actions(partial(count_successor_intervals, **bound))

        @jax.jit
        def batch_count(state_min, state_max, inputs):
            num, span, union = count_fn(state_min, state_max, inputs)
            return jnp.max(num, axis=(0, 1)), jnp.max(span, axis=(0, 1)), jnp.max(union, axis=(0, 1))

        t = time.time()

        self.num_regions = len(partition.regions['lower_bounds'])
        self.num_actions = partition.regions['actions'].shape[1]
        S, A, D = self.num_regions, self.num_actions, partition.dimension
        # Choose the grid-index dtype up front (no after-the-fact conversion of these multi-GB arrays): int8
        # when every dimension has <= 127 cells, else int16. It must be signed: indices are stored *unclipped*
        # and can be negative (out-of-grid successors map to the absorbing state downstream).
        idx_dtype = np.int8 if int(np.max(partition.number_per_dim)) <= INT8_MAX else np.int16
        idx_info = np.iinfo(idx_dtype)

        # Process state regions in batches: each call handles a [batch, num_actions] computation
        # instead of one [num_actions] computation, reducing Python–JAX round trips by frs_batch_size.
        starts, ends = create_batches(self.num_regions, args.frs_batch_size)

        def batch_inputs(batch_start, batch_end):
            # The three loop-varying arguments for states [batch_start:batch_end]; the rest are bound.
            actions_slice = partition.regions['actions']
            # DensePartition stores actions as (1, num_actions, action_dim); broadcast to batch size.
            # SparsePartition stores (num_states, num_actions, action_dim); slice normally.
            if actions_slice.shape[0] == 1:
                actions_batch = jnp.broadcast_to(actions_slice, (batch_end - batch_start, *actions_slice.shape[1:]))
            else:
                actions_batch = actions_slice[batch_start:batch_end]
            return (partition.regions['lower_bounds'][batch_start:batch_end],
                    partition.regions['upper_bounds'][batch_start:batch_end],
                    actions_batch)

        # Pass 1: the number of merged intervals and the spans per dimension (maxima over all pairs), so the
        # final arrays can be allocated at their exact size up front. Pass 2 then writes every batch straight
        # into them, so no second copy of the (multi-GB) reachability data is ever resident.
        slots, max_span, union_span = np.ones(D, dtype=int), np.ones(D, dtype=int), np.ones(D, dtype=int)
        for batch_start, batch_end in tqdm(zip(starts, ends), total=len(starts), desc='FRS pass 1/2'):
            num, span, union = jax.device_get(batch_count(*batch_inputs(batch_start, batch_end)))
            np.maximum(slots, num, out=slots)
            np.maximum(max_span, span, out=max_span)
            np.maximum(union_span, union, out=union_span)

        self.slots = tuple(int(x) for x in slots)
        self.max_slice = tuple(int(x) for x in max_span)
        self.union_span = tuple(int(x) for x in union_span)
        num_boxes = int(np.prod(slots))

        intervals_fn = over_states_and_actions(
            partial(successor_intervals, **bound, noise_probs=noise_probs_dev, slots=self.slots))
        batch_intervals = jax.jit(intervals_fn)

        # Pass 2: the merged intervals and the successor boxes' probabilities. Every pair's probabilities sum
        # to the noise partition's total mass (no merged interval is cut off by the slots).
        noise_mass = float(np.prod([np.sum(p) for p in per_dim_probs]))
        self.interval_lb = np.zeros((S, A, int(slots.sum())), dtype=idx_dtype)
        self.interval_ub = np.zeros((S, A, int(slots.sum())), dtype=idx_dtype)
        self.box_probs = np.zeros((S, A, num_boxes), dtype=args.floatprecision)
        for batch_start, batch_end in tqdm(zip(starts, ends), total=len(starts), desc='FRS pass 2/2'):
            lb, ub, probs = jax.device_get(batch_intervals(*batch_inputs(batch_start, batch_end)))
            # Indices may run slightly outside [0, number_per_dim) (OOB successors). Cheap per-batch
            # guard so an unexpected out-of-range index fails loudly instead of silently wrapping.
            if int(lb.min()) < idx_info.min or int(ub.max()) > idx_info.max:
                raise OverflowError(
                    f"FRS grid index out of {np.dtype(idx_dtype).name} range in batch [{batch_start}:{batch_end}] "
                    f"(min {int(lb.min())}, max {int(ub.max())})."
                )
            assert np.allclose(probs.sum(axis=-1), noise_mass, atol=1e-5)
            self.interval_lb[batch_start:batch_end] = lb
            self.interval_ub[batch_start:batch_end] = ub
            self.box_probs[batch_start:batch_end] = probs

        logger.info(f"- FRS index intervals stored as {np.dtype(idx_dtype).name}")
        logger.info(f"- Maximum span of the forward reachable sets: {self.max_slice}")
        logger.info(f"- Successor intervals per dimension: {self.slots} -> {num_boxes} successor boxes per "
                    f"state-action; span of their union: {self.union_span}")
        logger.info(f'- Forward reachable sets computed (took {(time.time() - t):.3f} sec.)')

        self.id = np.arange(self.num_actions)

        logger.info(f'Time for reachability computations: {(time.time() - t_total):.3f} sec.')
