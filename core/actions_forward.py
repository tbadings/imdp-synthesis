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
    Computes the forward reachable set given a set of input parameters.

    :param step_set: Function that computes the minimum and maximum reachable states given the state bounds and input.
    :param state_min: Lower bound of the box (of states ) to propagate.
    :param state_max: Upper bound of the box (of states ) to propagate.
    :param input: Control input for the dynamical system.
    :param cov_diag: Diagonal entries of the covariance matrix
    :param number_per_dim: The number of cells per dimension in the state space grid.
    :param cell_width: The width of cells along each dimension.
    :param boundary_lb: The lower bound of the grid of the state space.
    :param boundary_ub: The upper bound of the grid of the state space.
    :return: A tuple containing:
        - frs_min: The minimum bound of the forward reachable set.
        - frs_max: The maximum bound of the forward reachable set.
        - frs_span: The number of grid cells encompassed by the forward reachable set.
        - idx_low: The lower index bounds in the grid corresponding to the forward reachable set.
        - idx_upp: The upper index bounds in the grid corresponding to the forward reachable set.
    """

    epsilon = 0.00

    frs_min, frs_max = step_set(state_min, state_max, input - epsilon, input + epsilon)

    # If covariance is zero, then the span equals the number of cells the forward reachable set contains at most
    frs_span = jnp.astype(jnp.ceil((frs_max - frs_min) / cell_width), int)

    state_min_norm = (frs_min - boundary_lb) / (boundary_ub - boundary_lb) * number_per_dim
    lb_contained_in = state_min_norm // 1

    idx_low = (jnp.clip(lb_contained_in, 0, (number_per_dim - 1)) * (cov_diag == 0)).astype(int)
    idx_upp = (jnp.clip(lb_contained_in + frs_span - 1, 0, number_per_dim - 1) * (cov_diag == 0) + (number_per_dim - 1) * (cov_diag != 0)).astype(int)

    return frs_min, frs_max, frs_span, idx_low, idx_upp


class RectangularForward(object):

    def __init__(self, partition, model, x_dims, u_dims):
        print('Define target points and forward reachable sets...')
        t_total = time.time()

        # Vectorized function over different sets of points
        vmap_forward_reach = jax.jit(jax.vmap(forward_reach, in_axes=(None, None, None, 0, None, None, None, None, None), out_axes=(0, 0, 0, 0, 0,)),
                                     static_argnums=(0))

        discrete_per_dimension = [np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i]) for i in u_dims]
        self.inputs = jnp.array(list(itertools.product(*discrete_per_dimension)))

        t = time.time()

        # Pad the omitted dimensions with zeros (to use the step_set function of the full model)
        inputs = jnp.zeros((len(self.inputs), model.p))
        inputs = inputs.at[:, u_dims].set(self.inputs)
        lower_bounds = jnp.zeros((len(partition.regions['lower_bounds']), model.n))
        upper_bounds = jnp.zeros((len(partition.regions['upper_bounds']), model.n))
        lower_bounds = lower_bounds.at[:, x_dims].set(partition.regions['lower_bounds'])
        upper_bounds = upper_bounds.at[:, x_dims].set(partition.regions['upper_bounds'])
        # Also pad the partition parameters with zeros (to use the step_set function of the full model)
        number_per_dim = jnp.ones(model.n)
        number_per_dim = number_per_dim.at[x_dims].set(partition.number_per_dim)
        cell_width = jnp.ones(model.n)
        cell_width = cell_width.at[x_dims].set(partition.cell_width)
        boundary_lb = jnp.full(model.n, -0.5)
        boundary_lb = boundary_lb.at[x_dims].set(partition.boundary_lb)
        boundary_ub = jnp.full(model.n, 0.5)
        boundary_ub = boundary_ub.at[x_dims].set(partition.boundary_ub)

        pbar = tqdm(enumerate(zip(lower_bounds, upper_bounds)), total=len(lower_bounds))
        self.max_slice = jnp.zeros(partition.dimension)
        
        # Pre-compute all inputs on device
        # discrete_inputs_jax = jax.device_put(self.inputs)
        # noise_cov = jax.device_put(model.noise['cov_diag'])
        # number_per_dim = jax.device_put(partition.number_per_dim)
        # cell_width = jax.device_put(partition.cell_width)
        # boundary_lb = jax.device_put(partition.boundary_lb)
        # boundary_ub = jax.device_put(partition.boundary_ub)
                
        self.frs_lb = np.zeros((len(lower_bounds), len(inputs), partition.dimension))
        self.frs_ub = np.zeros_like(self.frs_lb)
        self.frs_idx_lb = np.zeros_like(self.frs_lb)
        self.frs_idx_ub = np.zeros_like(self.frs_lb)

        for i, (lb, ub) in pbar:
            # Batch compute forward reachable sets for all actions
            flb, fub, _, fil, fiu = vmap_forward_reach(
                    model.step_set, 
                    lb, 
                    ub, 
                    inputs, 
                    model.noise['cov_diag'], 
                    number_per_dim, 
                    cell_width, 
                    boundary_lb, 
                    boundary_ub)

            if len(x_dims) != model.n:
                self.frs_lb[i] = flb[:, x_dims]
                self.frs_ub[i] = fub[:, x_dims]
                self.frs_idx_lb[i] = fil[:, x_dims]
                self.frs_idx_ub[i] = fiu[:, x_dims]
            else:
                self.frs_lb[i] = flb
                self.frs_ub[i] = fub
                self.frs_idx_lb[i] = fil
                self.frs_idx_ub[i] = fiu
                
        # Compute max_slice after all forward reachable sets are computed
        self.max_slice = jnp.max(self.frs_idx_ub - self.frs_idx_lb + 1, axis=(0, 1))
        self.max_slice = tuple(np.astype(np.array(self.max_slice), int).tolist())

        print(f'- Forward reachable sets computed (took {(time.time() - t):.3f} sec.)')

        self.idxs = np.arange(len(self.inputs))

        print(f'Defining actions took {(time.time() - t_total):.3f} sec.')
        print('')
        return