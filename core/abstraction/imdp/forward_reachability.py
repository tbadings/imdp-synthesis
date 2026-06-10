import itertools
import logging
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from core.utils import create_batches

logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnums=(0))
def forward_reach(step_set, state_min, state_max, input, state_wrap, support_radius, number_per_dim, cell_width, boundary_lb, boundary_ub, shrink_frs):
    """
    Computes the forward reachable set for a given state region and control input.

    This function propagates a box-shaped state region forward in time using the dynamical system's
    step function. It computes both the continuous bounds and the discrete grid indices of the
    resulting forward reachable set.

    :param step_set: Function that computes the minimum and maximum reachable states given the 
                     state bounds and input. Signature: (state_min, state_max, input_min, input_max) -> (next_min, next_max)
    :param state_min: Lower bound of the state box to propagate (shape: [state_dim])
    :param state_max: Upper bound of the state box to propagate (shape: [state_dim])
    :param input: Control input for the dynamical system (shape: [input_dim])
    :param state_wrap: Boolean indicating whether the state space is wrapped (shape: [state_dim])
    :param support_radius: Radius of the support of the noise distribution (shape: [state_dim])
    :param number_per_dim: Number of grid cells per dimension in the state space discretization (shape: [state_dim])
    :param cell_width: Width of grid cells along each dimension (shape: [state_dim])
    :param boundary_lb: Lower bound of the state space grid (shape: [state_dim])
    :param boundary_ub: Upper bound of the state space grid (shape: [state_dim])
    :return: Tuple containing:
        - frs_min: Continuous lower bound of the forward reachable set (shape: [state_dim])
        - frs_max: Continuous upper bound of the forward reachable set (shape: [state_dim])
        - frs_span: Number of grid cells spanned by the forward reachable set per dimension (shape: [state_dim])
        - idx_low: Lower grid index bounds of the forward reachable set (shape: [state_dim])
        - idx_upp: Upper grid index bounds of the forward reachable set (shape: [state_dim])
    """

    # Small epsilon for numerical stability (currently set to zero)
    epsilon = 0.0

    # Compute the continuous bounds of the forward reachable set
    frs_min, frs_max = step_set(state_min, state_max, input - epsilon, input + epsilon)

    # Shrink frs bounds slightly for numerical stability (avoids issues when frs lands exactly on a cell boundary)
    frs_min = frs_min + shrink_frs
    frs_max = frs_max - shrink_frs

    frs_min_plus_noise = frs_min - support_radius
    frs_max_plus_noise = frs_max + support_radius

    # Calculate how many grid cells the forward reachable set spans in each dimension
    # Note: When covariance is zero, this gives the exact discrete span
    # The +1 is necessary to get correct upper bounds when the lower bound is just below a grid boundary 
    # (e.g., cell width of 1, lower bound of 0.8, upper bound of 2.2 spans not 2 but 3 cells)
    frs_span = jnp.astype(jnp.ceil((frs_max_plus_noise - frs_min_plus_noise) / cell_width) + 1, int)

    # Normalize the minimum bound to grid coordinates
    state_min_norm = (frs_min_plus_noise - boundary_lb) / (boundary_ub - boundary_lb) * number_per_dim
    lb_contained_in = state_min_norm // 1

    # Compute lower grid indices (clipped to valid range)
    # For dimensions with noise (cov_diag != 0), the index is set to 0
    idx_low = (jnp.clip(lb_contained_in, 0, (number_per_dim - 1)) * (~state_wrap)).astype(int)
    
    # Compute upper grid indices (clipped to valid range)
    # For dimensions with noise (cov_diag != 0), the index spans the entire dimension
    idx_upp = (jnp.clip(lb_contained_in + frs_span - 1, 0, number_per_dim - 1) * (~state_wrap) + (number_per_dim - 1) * (state_wrap)).astype(int)

    return frs_min, frs_max, frs_span, idx_low, idx_upp


class RectangularForward(object):
    """
    Computes and stores forward reachable sets for a rectangular partition of the state space.

    This class pre-computes the forward reachable sets for all state regions in a partition
    and all discrete control actions. The results are stored for efficient lookup during
    dynamic programming or reachability analysis.

    Attributes:
        inputs (jnp.ndarray): Discrete control actions, shape [num_actions, input_dim]
        frs_lb (np.ndarray): Lower bounds of forward reachable sets, shape [num_regions, num_actions, state_dim]
        frs_ub (np.ndarray): Upper bounds of forward reachable sets, shape [num_regions, num_actions, state_dim]
        frs_idx_lb (np.ndarray): Lower grid indices of forward reachable sets, shape [num_regions, num_actions, state_dim], dtype int16
        max_slice (tuple): Maximum span of forward reachable sets across all regions and actions per dimension
        idxs (np.ndarray): Indices of all actions, shape [num_actions]
    """

    def __init__(self, args, partition, model):
        """
        Initialize and compute forward reachable sets for all regions and actions.

        :param partition: Partition object containing the discretized state space
        :param model: Model object containing the dynamics and control action specifications
        """
        logger.info('Define target points and forward reachable sets...')
        t_total = time.time()

        # Inner vmap over control actions, outer vmap over a batch of state regions.
        # This reduces Python–JAX round trips from num_regions to ceil(num_regions / frs_batch_size).
        vmap_over_actions = jax.vmap(
            forward_reach,
            in_axes=(None, None, None, 0, None, None, None, None, None, None, None),
            out_axes=(0, 0, 0, 0, 0),
        )
        batch_forward_reach = jax.jit(
            jax.vmap(
                vmap_over_actions,
                in_axes=(None, 0, 0, 0, None, None, None, None, None, None, None),
                out_axes=(0, 0, 0, 0, 0),
            ),
            static_argnums=(0),
        )

        t = time.time()

        # Allocate output arrays
        self.num_regions = len(partition.regions['lower_bounds'])
        self.num_actions = partition.regions['actions'].shape[1]
        self.frs_lb = np.zeros((self.num_regions, self.num_actions, partition.dimension), dtype=args.floatprecision)
        self.frs_ub = np.zeros_like(self.frs_lb)
        self.frs_idx_lb = np.zeros((self.num_regions, self.num_actions, partition.dimension), dtype=np.int16)
        # max_slice is computed incrementally per batch to avoid storing frs_idx_ub
        max_span = np.zeros(partition.dimension, dtype=int)

        # Pre-load shared (non-batched) tensors to device once to avoid repeated transfers
        wrap_dev = jax.device_put(model.wrap)
        support_radius_dev = jax.device_put(model.noise['support_radius'])
        npd_dev = jax.device_put(partition.number_per_dim)
        cw_dev = jax.device_put(partition.cell_width)
        blb_dev = jax.device_put(partition.boundary_lb)
        bub_dev = jax.device_put(partition.boundary_ub)

        # Process state regions in batches: each call handles a [batch, num_actions] computation
        # instead of one [num_actions] computation, reducing Python–JAX round trips by frs_batch_size.
        starts, ends = create_batches(self.num_regions, args.frs_batch_size)
        pbar = tqdm(zip(starts, ends), total=len(starts))
        for batch_start, batch_end in pbar:
            batch_size = batch_end - batch_start
            actions_slice = partition.regions['actions']
            # RectangularPartition stores actions as (1, num_actions, action_dim); broadcast to batch size.
            # SparsePartition stores (num_states, num_actions, action_dim); slice normally.
            if actions_slice.shape[0] == 1:
                actions_batch = jnp.broadcast_to(actions_slice, (batch_size, *actions_slice.shape[1:]))
            else:
                actions_batch = actions_slice[batch_start:batch_end]
            flb, fub, _, fil, fiu = batch_forward_reach(
                model.step_set,
                partition.regions['lower_bounds'][batch_start:batch_end],
                partition.regions['upper_bounds'][batch_start:batch_end],
                actions_batch,
                wrap_dev,
                support_radius_dev,
                npd_dev,
                cw_dev,
                blb_dev,
                bub_dev,
                args.shrink_frs,
            )
            flb, fub, fil, fiu = jax.device_get((flb, fub, fil, fiu))
            self.frs_lb[batch_start:batch_end] = flb
            self.frs_ub[batch_start:batch_end] = fub
            self.frs_idx_lb[batch_start:batch_end] = fil.astype(np.int16)
            # Update max span incrementally to avoid storing full frs_idx_ub array
            batch_span = fiu - fil + 1
            np.maximum(max_span, np.max(batch_span, axis=(0, 1)).astype(int), out=max_span)

        # Store the maximum span of forward reachable sets
        # This is used to allocate sufficient memory for transition probability computations
        self.max_slice = tuple(max_span.tolist())
        logger.info(f"- Max state-slice to compute probability intervals over (including noise): {self.max_slice}")
        logger.info(f'- Forward reachable sets computed (took {(time.time() - t):.3f} sec.)')

        self.id = np.arange(self.num_actions)

        logger.info(f'Defining actions took {(time.time() - t_total):.3f} sec.')
        
        return
