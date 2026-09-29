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

def forward_reach_noise(state_min, state_max, input, step_set, cell_width, boundary_lb, shrink_frs,
                        noise_lb, noise_ub, noise_probs, num_boxes):
    """
    Computes the forward reachable set for a given state region and control input.

    This function propagates a box-shaped state region forward in time using the dynamical system's
    step function. It computes both the continuous bounds and the discrete grid indices of the
    resulting forward reachable set.

    The first three arguments (state_min, state_max, input) are the only ones that vary across the
    state regions / actions loop; the remaining arguments are constant and are intended to be bound
    once via functools.partial before vmapping (see RectangularForward).

    :param state_min: Lower bound of the state box to propagate (shape: [state_dim])
    :param state_max: Upper bound of the state box to propagate (shape: [state_dim])
    :param input: Control input for the dynamical system (shape: [input_dim])
    :param step_set: Function that computes the minimum and maximum reachable states given the
                     state bounds and input. Signature: (state_min, state_max, input_min, input_max) -> (next_min, next_max)
    :param cell_width: Width of grid cells along each dimension (shape: [state_dim])
    :param boundary_lb: Lower bound of the state space grid (shape: [state_dim])
    :param shrink_frs: Amount to shrink the forward reachable set by on each side (scalar)
    :param noise_lb: Per-dimension lower bounds of the noise intervals, in ascending order (tuple of
                     state_dim arrays, shape [n_d] each). The noise cells are all combinations of one
                     interval per dimension (num_noise_cells = prod_d n_d).
    :param noise_ub: Per-dimension upper bounds of the noise intervals (tuple of state_dim arrays, shape [n_d] each)
    :param noise_probs: Per-dimension probability mass of the noise intervals; a noise cell's
                        probability is the product over dimensions (tuple of state_dim arrays, shape [n_d] each)
    :param num_boxes: Static number of rows of the returned arrays; an upper bound on the number of
                      merged entries (see RectangularForward)
    :return: Tuple of arrays, each with a leading axis of size num_boxes. Noise cells whose
        successor box (idx_lb, idx_ub) is identical are merged into a single entry; the unique
        entries are packed at the top of each array and the remaining rows are inactive padding
        (probability 0). Entries are:
        - frs_span: Number of grid cells spanned by the forward reachable set per dimension (shape: [num_boxes, state_dim])
        - idx_lb: Lower grid index bounds of the forward reachable set (shape: [num_boxes, state_dim])
        - idx_ub: Upper grid index bounds of the forward reachable set (shape: [num_boxes, state_dim])
        - probs: Probability mass of each (merged) entry; merged cells' probabilities are summed and
                 inactive padding entries are 0 (shape: [num_boxes])
        - num_active: Number of unique (active) entries, i.e. how many leading rows of the arrays
                      above are populated; the remaining num_boxes - num_active rows are padding (scalar)
    """

    # Continuous bounds of the (noise-free) forward reachable set, shrunk slightly for numerical
    # stability (avoids issues when the FRS lands exactly on a cell boundary). epsilon is currently 0.
    epsilon = 0.0
    frs_min, frs_max = step_set(state_min, state_max, input - epsilon, input + epsilon)
    frs_min = frs_min + shrink_frs
    frs_max = frs_max - shrink_frs

    # --- Merge noise cells that map to the same successor box ------------------------------
    # A noise cell's successor box is the product of one grid-index interval per dimension, and each
    # interval depends only on the cell's noise interval in that dimension. So two cells share a box
    # iff they share its interval in every dimension: merge each dimension on its own, then compose
    # the merged intervals into boxes (probabilities multiply, as the noise dimensions are independent).
    merged_lb, merged_ub, merged_probs, num_merged = [], [], [], []
    for d in range(len(noise_lb)):
        # Grid cell index containing each FRS bound after shifting by every noise interval of this
        # dimension. Indices are left unclipped: they may fall outside [0, number_per_dim - 1], which
        # is resolved downstream (out-of-grid successors map to the absorbing state, wrapped dims
        # taken modulo).
        lb = jnp.floor((frs_min[d] + noise_lb[d] - boundary_lb[d]) / cell_width[d]).astype(int)
        ub = jnp.floor((frs_max[d] + noise_ub[d] - boundary_lb[d]) / cell_width[d]).astype(int)

        if lb.shape[0] == 1:
            # A single noise interval (e.g. a dimension without noise) is always one merged interval.
            merged_lb.append(lb)
            merged_ub.append(ub)
            merged_probs.append(noise_probs[d])
            num_merged.append(1)
            continue

        # The noise intervals are ascending, so lb and ub are non-decreasing and equal (lb, ub) pairs
        # are adjacent: label the start of each run, pack the runs at the top and sum their probabilities.
        is_first = jnp.concatenate([jnp.array([True]), (lb[1:] != lb[:-1]) | (ub[1:] != ub[:-1])])
        slot = jnp.cumsum(is_first) - 1
        merged_lb.append(jnp.zeros_like(lb).at[slot].set(lb))
        merged_ub.append(jnp.zeros_like(ub).at[slot].set(ub))
        merged_probs.append(jnp.zeros_like(noise_probs[d]).at[slot].add(noise_probs[d]))
        num_merged.append(is_first.sum())

    # Compose the merged intervals into boxes. The output has num_boxes rows (static shape, so the
    # function stays jit/vmap-able): row i combines interval j_d of every dimension d, where
    # (j_0, ..., j_{D-1}) are the mixed-radix digits of i in base (num_merged_0, ..., num_merged_{D-1}).
    # So the prod_d num_merged_d boxes are packed at the top and the remaining rows are inactive
    # padding (index 0, probability 0). A dimension with a single noise interval always has digit 0.
    row = jnp.arange(num_boxes)
    digits, stride = [None] * len(num_merged), jnp.ones((), row.dtype)
    for d in reversed(range(len(num_merged))):
        digits[d] = jnp.zeros_like(row) if noise_lb[d].shape[0] == 1 else (row // stride) % num_merged[d]
        stride = stride * num_merged[d]
    num_active = stride                                                      # prod_d num_merged_d
    active = row < num_active                                                # (num_boxes,)

    idx_lb = jnp.stack([x[digits[d]] for d, x in enumerate(merged_lb)], axis=1) * active[:, None]
    idx_ub = jnp.stack([x[digits[d]] for d, x in enumerate(merged_ub)], axis=1) * active[:, None]
    probs = jnp.prod(jnp.stack([x[digits[d]] for d, x in enumerate(merged_probs)], axis=1), axis=1) * active

    # Number of grid cells each (merged) forward reachable set spans per dimension.
    frs_span = idx_ub - idx_lb + 1

    return frs_span, idx_lb, idx_ub, probs, num_active

def count_merged_boxes(state_min, state_max, input, step_set, cell_width, boundary_lb, shrink_frs,
                       noise_lb, noise_ub):
    """
    Number of merged forward reachable sets (num_active) and their maximum span per dimension, as
    forward_reach_noise computes them, but without composing the boxes. Used as a cheap first pass
    to size the stored arrays exactly (see RectangularForward), so it must use the same arithmetic.

    The arguments are those of forward_reach_noise (noise_probs and num_boxes are not needed).

    :return: Tuple (num_active, max_span): the number of merged entries (scalar) and the maximum
        number of grid cells a merged forward reachable set spans per dimension (shape: [state_dim])
    """
    epsilon = 0.0
    frs_min, frs_max = step_set(state_min, state_max, input - epsilon, input + epsilon)
    frs_min = frs_min + shrink_frs
    frs_max = frs_max - shrink_frs

    num_merged, max_span = [], []
    for d in range(len(noise_lb)):
        lb = jnp.floor((frs_min[d] + noise_lb[d] - boundary_lb[d]) / cell_width[d]).astype(int)
        ub = jnp.floor((frs_max[d] + noise_ub[d] - boundary_lb[d]) / cell_width[d]).astype(int)
        # The merged intervals are the runs of equal (lb, ub) pairs, so they have the same spans.
        num_merged.append(1 + jnp.sum((lb[1:] != lb[:-1]) | (ub[1:] != ub[:-1])))
        max_span.append(jnp.max(ub - lb + 1))

    return jnp.prod(jnp.stack(num_merged)), jnp.stack(max_span)

class RectangularForward(object):
    """
    Computes and stores forward reachable sets for a rectangular partition of the state space.

    This class pre-computes the forward reachable sets for all state regions in a partition
    and all discrete control actions. The results are stored for efficient lookup during
    dynamic programming or reachability analysis.

    For SVMDP, one (merged) forward reachable set is produced per noise cell: noise cells that map
    to the same successor box are merged, so per (state, action) only the leading num_active entries
    are populated and the remaining max_active_noise_cells - num_active entries are inactive padding.

    Attributes:
        frs_idx_lb (np.ndarray): Lower grid indices of forward reachable sets,
            shape [num_regions, num_actions, max_active_noise_cells, state_dim], dtype int8 or int16
        frs_idx_ub (np.ndarray): Upper grid indices of forward reachable sets,
            shape [num_regions, num_actions, max_active_noise_cells, state_dim], dtype int8 or int16
        frs_noise_probs (np.ndarray): Probability mass of each merged entry (merged cells summed; padding 0),
            shape [num_regions, num_actions, max_active_noise_cells]
        frs_noise_num_active (np.ndarray): Number of populated (merged) entries per (state, action),
            shape [num_regions, num_actions], dtype int32
        max_slice (tuple): Maximum span of forward reachable sets across all regions and actions per dimension
        max_active_noise_cells (int): Maximum number of active (merged) noise-cell entries across all (state, action) pairs
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
        # forward_reach_noise merges each dimension on its own and composes the merged
        # intervals into one (merged) forward reachable set per distinct successor box.
        per_dim_cells = [np.asarray(c) for c in model.noise.partition['per_dim_cells']]  # D x (n_d, 2)
        per_dim_probs = [np.asarray(p) for p in model.noise.partition['per_dim_probs']]  # D x (n_d,)
        # The per-dimension merge only compares neighbouring intervals, so they must be ascending.
        assert all(np.all(np.diff(c[:, 0]) > 0) for c in per_dim_cells)
        noise_lb_dev = tuple(jax.device_put(c[:, 0]) for c in per_dim_cells)
        noise_ub_dev = tuple(jax.device_put(c[:, 1]) for c in per_dim_cells)
        noise_probs_dev = tuple(jax.device_put(p) for p in per_dim_probs)

        # Pre-load shared (non-batched) tensors to device once to avoid repeated transfers.
        cw_dev = jax.device_put(partition.cell_width)
        blb_dev = jax.device_put(partition.boundary_lb)

        # Static number of rows forward_reach_noise returns: an upper bound on the number of merged
        # boxes, so the returned arrays stay close to their merged width instead of one row per noise
        # cell. In dimension d, the lower (upper) grid indices of the noise intervals lie in a window of
        # the noise spread, so they take at most floor(spread / cell_width) + 2 distinct values; as both
        # are non-decreasing, at most (#lower values) + (#upper values) - 1 intervals are distinct.
        # The 1e-4 absorbs float rounding when the spread is an exact multiple of the cell width.
        cell_width = np.asarray(partition.cell_width)
        num_boxes = 1
        for d, c in enumerate(per_dim_cells):
            lb_values = int(np.floor(np.ptp(c[:, 0]) / cell_width[d] + 1e-4)) + 2
            ub_values = int(np.floor(np.ptp(c[:, 1]) / cell_width[d] + 1e-4)) + 2
            num_boxes *= min(len(c), lb_values + ub_values - 1)

        # Bind the constant arguments once; only (state_min, state_max, input) vary in the loop.
        frs_fn = partial(
            forward_reach_noise,
            step_set=model.step_set,
            cell_width=cw_dev,
            boundary_lb=blb_dev,
            shrink_frs=args.shrink_frs,
            noise_lb=noise_lb_dev,
            noise_ub=noise_ub_dev,
            noise_probs=noise_probs_dev,
            num_boxes=num_boxes,
        )

        # Inner vmap over control actions, outer vmap over a batch of state regions; only the three
        # varying arguments are mapped. This reduces Python–JAX round trips from num_regions to
        # ceil(num_regions / frs_batch_size).
        vmap_over_actions = jax.vmap(frs_fn, in_axes=(None, None, 0))
        vmap_over_states = jax.vmap(vmap_over_actions, in_axes=(0, 0, 0))

        @jax.jit
        def batch_forward_reach(state_min, state_max, inputs):
            # The first pass provides max_slice, so the (full-size) span output is dropped; XLA then
            # never materialises it.
            _, idx_lb, idx_ub, probs, num_active = vmap_over_states(state_min, state_max, inputs)
            return idx_lb, idx_ub, probs, num_active

        # First pass: only the number of merged entries and the span per (state, action).
        count_fn = partial(
            count_merged_boxes,
            step_set=model.step_set,
            cell_width=cw_dev,
            boundary_lb=blb_dev,
            shrink_frs=args.shrink_frs,
            noise_lb=noise_lb_dev,
            noise_ub=noise_ub_dev,
        )
        vmap_count = jax.vmap(jax.vmap(count_fn, in_axes=(None, None, 0)), in_axes=(0, 0, 0))

        @jax.jit
        def batch_count(state_min, state_max, inputs):
            num_active, span = vmap_count(state_min, state_max, inputs)
            return num_active, jnp.max(span, axis=(0, 1))

        t = time.time()

        self.num_regions = len(partition.regions['lower_bounds'])
        self.num_actions = partition.regions['actions'].shape[1]
        S, A, D = self.num_regions, self.num_actions, partition.dimension
        # Choose the grid-index dtype up front (no after-the-fact conversion of these multi-GB arrays):
        # int8 when every dimension has <= 127 cells, else int16. int8 halves the footprint of these
        # dominant [S, A, C, D] arrays for fine-grained models like Drone6D. It must be int8, not uint8:
        # indices are stored *unclipped* and can be negative (out-of-grid successors map to the absorbing
        # state downstream), and box_to_ids_single enumerates `arange(span) + idx_lb` masking `col < 0`,
        # so the sign must be preserved. Models with a dimension > 127 cells (e.g. MountainCar) keep int16.
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

        # Pass 1: the number of merged entries per (state, action) and the maximum span, so the final
        # arrays can be allocated at their exact size up front. Pass 2 then writes every batch straight
        # into them, so no second copy of the (multi-GB) reachability data is ever resident.
        self.frs_noise_num_active = np.zeros((S, A), dtype=np.int32)
        max_span = np.zeros(D, dtype=int)
        for batch_start, batch_end in tqdm(zip(starts, ends), total=len(starts), desc='FRS pass 1/2'):
            num_active, batch_span = jax.device_get(batch_count(*batch_inputs(batch_start, batch_end)))
            self.frs_noise_num_active[batch_start:batch_end] = num_active
            np.maximum(max_span, batch_span, out=max_span)

        # TODO: With no wrap, max_span is potentially conservative (there may be many indices OOB that can already be ignored)

        # Store the maximum span of forward reachable sets
        # This is used to allocate sufficient memory for transition probability computations
        self.max_slice = tuple(max_span.tolist())
        self.max_active_noise_cells = int(self.frs_noise_num_active.max(initial=0))
        # num_boxes is a static bound on the number of merged entries; fail loudly if it was too small.
        assert self.max_active_noise_cells <= num_boxes, \
            f"{self.max_active_noise_cells} merged boxes exceed the bound num_boxes={num_boxes}"

        # Pass 2: per (state, action) the kernel returns num_boxes entries, of which merging leaves
        # only the leading num_active populated, so only the first K are stored.
        K = self.max_active_noise_cells
        self.frs_idx_lb = np.zeros((S, A, K, D), dtype=idx_dtype)
        self.frs_idx_ub = np.zeros((S, A, K, D), dtype=idx_dtype)
        self.frs_noise_probs = np.zeros((S, A, K), dtype=args.floatprecision)
        for batch_start, batch_end in tqdm(zip(starts, ends), total=len(starts), desc='FRS pass 2/2'):
            # Only the three loop-varying arguments are passed; the rest are bound in frs_fn.
            frs_lb, frs_ub, frs_prob, frs_nact = batch_forward_reach(*batch_inputs(batch_start, batch_end))
            # JAX dispatches asynchronously; block so the timing reflects actual compute.
            jax.block_until_ready((frs_lb, frs_ub, frs_prob, frs_nact))

            frs_lb, frs_ub, frs_prob, frs_nact = jax.device_get((frs_lb, frs_ub, frs_prob, frs_nact))
            # Indices may run slightly outside [0, number_per_dim) (OOB successors). Cheap per-batch
            # guard so an unexpected out-of-range index fails loudly instead of silently wrapping.
            if int(frs_lb.min()) < idx_info.min or int(frs_ub.max()) > idx_info.max:
                raise OverflowError(
                    f"FRS grid index out of {np.dtype(idx_dtype).name} range in batch [{batch_start}:{batch_end}] "
                    f"(min {int(frs_lb.min())}, max {int(frs_ub.max())})."
                )
            # Both passes count the same merged entries, and the entries beyond K are empty padding.
            assert np.array_equal(frs_nact, self.frs_noise_num_active[batch_start:batch_end])
            assert np.all(frs_prob[:, :, K:] == 0)
            self.frs_idx_lb[batch_start:batch_end] = frs_lb[:, :, :K]
            self.frs_idx_ub[batch_start:batch_end] = frs_ub[:, :, :K]
            self.frs_noise_probs[batch_start:batch_end] = frs_prob[:, :, :K]
        logger.info(f"- FRS index boxes stored as {np.dtype(idx_dtype).name}")

        logger.info(f"- Maximum span of the forward reachable sets: {self.max_slice}")
        logger.info(f"- Max number of noise cells state-slice after merging: {self.max_active_noise_cells}")
        logger.info(f'- Forward reachable sets computed (took {(time.time() - t):.3f} sec.)')

        self.id = np.arange(self.num_actions)

        logger.info(f'Time for reachability computations: {(time.time() - t_total):.3f} sec.')

        # The successor cell IDs spanned by each box are NOT materialised here: that array has
        # shape [S, A, max_active_noise_cells, prod(max_span)] and is tens of GB for 3-D models.
        # Instead the DP recomposes them on the fly from the compact boxes (frs_idx_lb/frs_idx_ub)
        # via core.abstraction.svmdp.successor_ids.box_to_ids_single (separable linear key).
