import itertools
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


@partial(jax.jit, static_argnums=(0))
def forward_reach(step_set, state_min, state_max, input, cov_diag, number_per_dim, cell_width, boundary_lb, boundary_ub):
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
    :param cov_diag: Diagonal entries of the noise covariance matrix (shape: [state_dim])
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

    # Calculate how many grid cells the forward reachable set spans in each dimension
    # Note: When covariance is zero, this gives the exact discrete span
    # The +1 is necessary to get correct upper bounds when the lower bound is just below a grid boundary 
    # (e.g., cell width of 1, lower bound of 0.8, upper bound of 2.2 spans not 2 but 3 cells)
    frs_span = jnp.astype(jnp.ceil((frs_max - frs_min) / cell_width) + 1, int)

    # Normalize the minimum bound to grid coordinates
    state_min_norm = (frs_min - boundary_lb) / (boundary_ub - boundary_lb) * number_per_dim
    lb_contained_in = state_min_norm // 1

    # TODO: Make a rigorous implementation of how to handle noise here. The current implementation is a heuristic that expands the reachable set by a fixed number of cells in dimensions with noise. A more principled approach would consider the actual distribution of the noise and how it affects the reachable set.

    # Compute lower grid indices (clipped to valid range)
    # For dimensions with noise (cov_diag != 0), the index is set to 0
    idx_low = (jnp.clip(lb_contained_in, 0, (number_per_dim - 1)) * (cov_diag == 0)).astype(int)
    
    # Compute upper grid indices (clipped to valid range)
    # For dimensions with noise (cov_diag != 0), the index spans the entire dimension
    idx_upp = (jnp.clip(lb_contained_in + frs_span - 1, 0, number_per_dim - 1) * (cov_diag == 0) + (number_per_dim - 1) * (cov_diag != 0)).astype(int)

    '''
    # Compute lower grid indices (clipped to valid range)
    # For dimensions with noise (cov_diag != 0), the index is set to 0
    q = 5 # Maximum number of cells the noise can "add" to the span of the reachable set
    idx_low = (jnp.clip(lb_contained_in, 0, (number_per_dim - 1)) - q * (cov_diag != 0)).astype(int)
    
    # Compute upper grid indices (clipped to valid range)
    # For dimensions with noise (cov_diag != 0), the index spans the entire dimension
    idx_upp = (jnp.clip(lb_contained_in + frs_span - 1, 0, number_per_dim - 1) + (2*q) * (cov_diag != 0)).astype(int)
    '''

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
        frs_idx_lb (np.ndarray): Lower grid indices of forward reachable sets, shape [num_regions, num_actions, state_dim]
        frs_idx_ub (np.ndarray): Upper grid indices of forward reachable sets, shape [num_regions, num_actions, state_dim]
        max_slice (tuple): Maximum span of forward reachable sets across all regions and actions per dimension
        idxs (np.ndarray): Indices of all actions, shape [num_actions]
    """

    def __init__(self, args, partition, model):
        """
        Initialize and compute forward reachable sets for all regions and actions.

        :param partition: Partition object containing the discretized state space
        :param model: Model object containing the dynamics and control action specifications
        """
        print('Define target points and forward reachable sets...')
        t_total = time.time()

        # Create vectorized function to compute forward reach for multiple actions in parallel
        # vmap over axis 0 (different actions), keeping other parameters fixed
        vmap_forward_reach = jax.jit(
            jax.vmap(
                forward_reach,
                in_axes=(None, None, None, 0, None, None, None, None, None),
                out_axes=(0, 0, 0, 0, 0)
            ),
            static_argnums=(0)
        )

        # Generate discrete action grid by taking Cartesian product of actions per dimension
        discrete_actions_per_dimension = [
            np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i])
            for i in range(len(model.num_actions))
        ]
        self.id_to_input = jnp.array(list(itertools.product(*discrete_actions_per_dimension)))

        t = time.time()

        # Initialize progress bar for iterating through all state regions
        pbar = tqdm(
            enumerate(zip(partition.regions['lower_bounds'], partition.regions['upper_bounds'])),
            total=len(partition.regions['lower_bounds'])
        )
        self.max_slice = jnp.zeros(partition.dimension)
        
        # Note: Pre-loading all inputs on device is possible but commented out
        # This could improve performance for very large action spaces
        # discrete_inputs_jax = jax.device_put(self.inputs)
        # noise_cov = jax.device_put(model.noise['cov_diag'])
        # number_per_dim = jax.device_put(partition.number_per_dim)
        # cell_width = jax.device_put(partition.cell_width)
        # boundary_lb = jax.device_put(partition.boundary_lb)
        # boundary_ub = jax.device_put(partition.boundary_ub)
        
        # Allocate storage for forward reachable set information
        num_regions = len(partition.regions['lower_bounds'])
        num_actions = len(self.id_to_input)
        self.frs_lb = np.zeros((num_regions, num_actions, partition.dimension), dtype=args.floatprecision)
        self.frs_ub = np.zeros_like(self.frs_lb, dtype=args.floatprecision)
        self.frs_idx_lb = np.zeros_like(self.frs_lb, dtype=args.floatprecision)
        self.frs_idx_ub = np.zeros_like(self.frs_lb, dtype=args.floatprecision)

        # Iterate through all state regions and compute forward reachable sets
        max_span = np.zeros(partition.dimension)
        for i, (lb, ub) in pbar:
            # Batch compute forward reachable sets for all actions using vectorized function
            flb, fub, _, fil, fiu = vmap_forward_reach(
                model.step_set,
                lb,
                ub,
                self.id_to_input,
                model.noise['cov_diag'],
                partition.number_per_dim,
                partition.cell_width,
                partition.boundary_lb,
                partition.boundary_ub
            )

            # Store the computed forward reachable set bounds and indices
            self.frs_lb[i] = np.array(flb)
            self.frs_ub[i] = np.array(fub)
            self.frs_idx_lb[i] = np.array(fil)
            self.frs_idx_ub[i] = np.array(fiu)
            
            # Incrementally update max span
            span = np.array(fiu) - np.array(fil) + 1
            max_span = np.maximum(max_span, np.max(span, axis=0))

        # Store the maximum span of forward reachable sets
        # This is used to allocate sufficient memory for transition probability computations
        self.max_slice = tuple(max_span.astype(int).tolist())

        print(f'- Forward reachable sets computed (took {(time.time() - t):.3f} sec.)')
        # Create array of action indices for efficient indexing        
        
        self.id = np.arange(len(self.id_to_input))

        print(f'Defining actions took {(time.time() - t_total):.3f} sec.')
        print('')
        return
