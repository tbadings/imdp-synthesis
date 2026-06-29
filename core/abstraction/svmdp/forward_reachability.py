import logging
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from core.utils import create_batches

logger = logging.getLogger(__name__)

def forward_reach_noise(state_min, state_max, input, step_set, cell_width, boundary_lb, shrink_frs,
                        noise_lb, noise_ub, noise_cells_probs):
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
    :param noise_lb: Lower bound of each noise cell (shape: [num_noise_cells, state_dim])
    :param noise_ub: Upper bound of each noise cell (shape: [num_noise_cells, state_dim])
    :param noise_cells_probs: Probability mass of each noise cell (shape: [num_noise_cells])
    :return: Tuple of arrays, each with a leading axis of size num_noise_cells. Noise cells whose
        successor box (idx_low, idx_upp) is identical are merged into a single entry; the unique
        entries are packed at the top of each array and the remaining rows are inactive padding
        (probability 0). Entries are:
        - frs_span: Number of grid cells spanned by the forward reachable set per dimension (shape: [num_noise_cells, state_dim])
        - idx_low: Lower grid index bounds of the forward reachable set (shape: [num_noise_cells, state_dim])
        - idx_upp: Upper grid index bounds of the forward reachable set (shape: [num_noise_cells, state_dim])
        - probs: Probability mass of each (merged) entry; merged cells' probabilities are summed and
                 inactive padding entries are 0 (shape: [num_noise_cells])
        - num_active: Number of unique (active) entries, i.e. how many leading rows of the arrays
                      above are populated; the remaining num_noise_cells - num_active rows are padding (scalar)
    """

    # Continuous bounds of the (noise-free) forward reachable set, shrunk slightly for numerical
    # stability (avoids issues when the FRS lands exactly on a cell boundary). epsilon is currently 0.
    epsilon = 0.0
    frs_min, frs_max = step_set(state_min, state_max, input - epsilon, input + epsilon)
    frs_min = frs_min + shrink_frs
    frs_max = frs_max - shrink_frs

    # Grid cell index containing each FRS bound after shifting by every noise cell's interval, one
    # row per noise cell. Broadcasting the (state_dim,) FRS bounds against the (num_noise_cells,
    # state_dim) noise bounds yields a leading num_noise_cells axis. Indices are left unclipped: they
    # may fall outside [0, number_per_dim - 1], which is resolved downstream (out-of-grid successors
    # map to the absorbing state, wrapped dims taken modulo).
    idx_low = jnp.floor((frs_min + noise_lb - boundary_lb) / cell_width).astype(int)
    idx_upp = jnp.floor((frs_max + noise_ub - boundary_lb) / cell_width).astype(int)

    # --- Merge noise cells that map to the same successor box ------------------------------
    # Noise cells whose (idx_low, idx_upp) grid box is identical describe the same successor set, so
    # they are merged into one entry whose probability is the sum of the merged cells' probabilities.
    # The output keeps the original num_noise_cells length (static shape, so the function stays
    # jit/vmap-able): unique entries are packed at the top and the remaining rows are inactive
    # padding (probability 0).
    N, D = idx_low.shape

    # Sort cells so identical successor boxes are adjacent, then label the start of each group.
    key = jnp.concatenate([idx_low, idx_upp], axis=1)                     # (N, 2D)
    perm = jnp.lexsort(tuple(key[:, c] for c in reversed(range(2 * D))))  # (N,)
    key_sorted = key[perm]                                                # (N, 2D)
    is_first = jnp.concatenate([
        jnp.array([True]),
        jnp.any(key_sorted[1:] != key_sorted[:-1], axis=1),
    ])                                                                    # (N,) start of each group
    slot = jnp.cumsum(is_first) - 1                                       # (N,) top-packed dest slot
    num_active = is_first.sum()                                           # scalar: number of unique entries

    # Scatter into top-packed slots: the box is identical within a group (a plain set is
    # well-defined and reuses the already-sorted key), and probabilities are summed. Untouched
    # padding slots keep their zero initialiser.
    key_merged = jnp.zeros((N, 2 * D), key.dtype).at[slot].set(key_sorted)
    idx_low, idx_upp = key_merged[:, :D], key_merged[:, D:]
    probs = jnp.zeros(N, noise_cells_probs.dtype).at[slot].add(noise_cells_probs[perm])

    # Number of grid cells each (merged) forward reachable set spans per dimension.
    frs_span = idx_upp - idx_low + 1

    return frs_span, idx_low, idx_upp, probs, num_active

class RectangularForward(object):
    """
    Computes and stores forward reachable sets for a rectangular partition of the state space.

    This class pre-computes the forward reachable sets for all state regions in a partition
    and all discrete control actions. The results are stored for efficient lookup during
    dynamic programming or reachability analysis.

    For SVMDP, one (merged) forward reachable set is produced per noise cell: noise cells that map
    to the same successor box are merged, so per (state, action) only the leading num_active entries
    are populated and the remaining num_noise_cells - num_active entries are inactive padding.

    Attributes:
        frs_idx_lb (np.ndarray): Lower grid indices of forward reachable sets,
            shape [num_regions, num_actions, num_noise_cells, state_dim], dtype int16
        frs_idx_ub (np.ndarray): Upper grid indices of forward reachable sets,
            shape [num_regions, num_actions, num_noise_cells, state_dim], dtype int16
        probs (np.ndarray): Probability mass of each merged entry (merged cells summed; padding 0),
            shape [num_regions, num_actions, num_noise_cells]
        num_active (np.ndarray): Number of populated (merged) entries per (state, action),
            shape [num_regions, num_actions], dtype int32
        max_slice (tuple): Maximum span of forward reachable sets across all regions and actions per dimension
        id (np.ndarray): Indices of all actions, shape [num_actions]
    """

    def __init__(self, args, partition, model):
        """
        Initialize and compute forward reachable sets for all regions and actions.

        :param partition: Partition object containing the discretized state space
        :param model: Model object containing the dynamics and control action specifications
        """
        logger.info('Define target points and forward reachable sets...')
        t_total = time.time()

        # Noise partition: forward_reach_noise produces one (merged) forward reachable set per
        # noise cell. cells with shape (C, D, 2) -> per-cell lower/upper bounds and probabilities.
        noise_cells = np.asarray(model.noise.partition['cells'])                    # (C, D, 2)
        noise_lb_dev = jax.device_put(noise_cells[:, :, 0])                         # (C, D)
        noise_ub_dev = jax.device_put(noise_cells[:, :, 1])                         # (C, D)
        noise_probs_dev = jax.device_put(np.asarray(model.noise.partition['probs']))  # (C,)

        # Pre-load shared (non-batched) tensors to device once to avoid repeated transfers.
        cw_dev = jax.device_put(partition.cell_width)
        blb_dev = jax.device_put(partition.boundary_lb)

        # Bind the constant arguments once; only (state_min, state_max, input) vary in the loop.
        frs_fn = partial(
            forward_reach_noise,
            step_set=model.step_set,
            cell_width=cw_dev,
            boundary_lb=blb_dev,
            shrink_frs=args.shrink_frs,
            noise_lb=noise_lb_dev,
            noise_ub=noise_ub_dev,
            noise_cells_probs=noise_probs_dev,
        )

        # Inner vmap over control actions, outer vmap over a batch of state regions; only the three
        # varying arguments are mapped. This reduces Python–JAX round trips from num_regions to
        # ceil(num_regions / frs_batch_size).
        vmap_over_actions = jax.vmap(frs_fn, in_axes=(None, None, 0))
        batch_forward_reach = jax.jit(jax.vmap(vmap_over_actions, in_axes=(0, 0, 0)))

        t = time.time()

        # Allocate output arrays. Per (state, action) the function returns C entries (one per noise
        # cell); cells mapping to the same successor box are merged, so only the leading num_active
        # entries of each (state, action) row are populated and the rest are inactive padding.
        self.num_regions = len(partition.regions['lower_bounds'])
        self.num_actions = partition.regions['actions'].shape[1]
        S, A, C, D = self.num_regions, self.num_actions, noise_cells.shape[0], partition.dimension
        self.frs_idx_lb = np.zeros((S, A, C, D), dtype=np.int16)
        self.frs_idx_ub = np.zeros((S, A, C, D), dtype=np.int16)
        self.frs_noise_probs = np.zeros((S, A, C), dtype=args.floatprecision)
        self.frs_noise_num_active = np.zeros((S, A), dtype=np.int32)
        # max_slice is computed incrementally per batch to avoid a second pass over the indices.
        
        max_span = np.zeros(D, dtype=int)
        max_active_noise_cells = 0

        # Process state regions in batches: each call handles a [batch, num_actions] computation
        # instead of one [num_actions] computation, reducing Python–JAX round trips by frs_batch_size.
        starts, ends = create_batches(self.num_regions, args.frs_batch_size)
        pbar = tqdm(zip(starts, ends), total=len(starts))
        for batch_start, batch_end in pbar:
            t = time.time()

            batch_size = batch_end - batch_start
            actions_slice = partition.regions['actions']
            # RectangularPartition stores actions as (1, num_actions, action_dim); broadcast to batch size.
            # SparsePartition stores (num_states, num_actions, action_dim); slice normally.
            if actions_slice.shape[0] == 1:
                actions_batch = jnp.broadcast_to(actions_slice, (batch_size, *actions_slice.shape[1:]))
            else:
                actions_batch = actions_slice[batch_start:batch_end]
            # Only the three loop-varying arguments are passed; the rest are bound in frs_fn.
            
            print(f"  Batch setup: {time.time() - t:.3f}s")
            t = time.time()

            fspan, fil, fiu, fprob, fnact = batch_forward_reach(
                partition.regions['lower_bounds'][batch_start:batch_end],
                partition.regions['upper_bounds'][batch_start:batch_end],
                actions_batch,
            )
            # JAX dispatches asynchronously; block so the timing reflects actual compute.
            jax.block_until_ready((fspan, fil, fiu, fprob, fnact))

            print(f"  Forward reach: {time.time() - t:.3f}s")
            t = time.time()

            fspan, fil, fiu, fprob, fnact = jax.device_get((fspan, fil, fiu, fprob, fnact))
            self.frs_idx_lb[batch_start:batch_end] = fil.astype(np.int16)
            self.frs_idx_ub[batch_start:batch_end] = fiu.astype(np.int16)
            self.frs_noise_probs[batch_start:batch_end] = fprob
            self.frs_noise_num_active[batch_start:batch_end] = fnact
            # Update max span incrementally (padding entries span 1 cell, so never inflate the max).
            np.maximum(max_span, np.max(fspan, axis=(0, 1, 2)).astype(int), out=max_span)
            max_active_noise_cells = np.maximum(max_active_noise_cells, np.max(fnact).astype(int))

            print(f"  Device get + store: {time.time() - t:.3f}s")

        # TODO: With no wrap, max_span is potentially conservative (there may be many indices OOB that can already be ignored)

        # Store the maximum span of forward reachable sets
        # This is used to allocate sufficient memory for transition probability computations
        self.max_slice = tuple(max_span.tolist())
        self.max_active_noise_cells = max_active_noise_cells

        # Remove noise cells that were merged (after a sanity check that we are only throwing away probability zero cells)
        assert np.all(self.frs_noise_probs[:, :, self.max_active_noise_cells:] == 0)
        self.frs_idx_lb = self.frs_idx_lb[:, :, :self.max_active_noise_cells, :]
        self.frs_idx_ub = self.frs_idx_ub[:, :, :self.max_active_noise_cells, :]
        self.frs_noise_probs = self.frs_noise_probs[:, :, :self.max_active_noise_cells]

        logger.info(f"- Maximum span of the forward reachable sets: {self.max_slice}")
        logger.info(f"- Max number of noise cells state-slice after merging: {self.max_active_noise_cells}")
        logger.info(f'- Forward reachable sets computed (took {(time.time() - t):.3f} sec.)')

        t = time.time()

        self.id = np.arange(self.num_actions)

        logger.info(f'Defining actions took {(time.time() - t_total):.3f} sec.')

        # The successor cell IDs spanned by each box are NOT materialised here: that array has
        # shape [S, A, max_active_noise_cells, prod(max_span)] and is tens of GB for 3-D models.
        # Instead the DP recomposes them on the fly from the compact boxes (frs_idx_lb/frs_idx_ub)
        # via core.abstraction.svmdp.successor_ids.box_to_ids_single (separable linear key).

        return