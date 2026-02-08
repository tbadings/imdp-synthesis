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

    def __init__(self, partition, model):
        print('Define target points and forward reachable sets...')
        t_total = time.time()

        # Vectorized function over different sets of points
        vmap_forward_reach = jax.vmap(forward_reach, in_axes=(None, None, None, 0, None, None, None, None, None), out_axes=(0, 0, 0, 0, 0,))

        discrete_per_dimension = [np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i]) for i in range(len(model.num_actions))]
        discrete_inputs = np.array(list(itertools.product(*discrete_per_dimension)))

        t = time.time()

        frs = {}
        pbar = tqdm(enumerate(zip(partition.regions['lower_bounds'], partition.regions['upper_bounds'])), total=len(partition.regions['lower_bounds']))
        self.max_slice = np.zeros(model.n)
        for i, (lb, ub) in pbar:
            # For every state, compute for every action the [lb,ub] forward reachable set
            flb, fub, fsp, fil, fiu = vmap_forward_reach(model.step_set, lb, ub, discrete_inputs, model.noise['cov_diag'], partition.number_per_dim, partition.cell_width,
                                                         partition.boundary_lb, partition.boundary_ub)

            frs[i] = {}
            frs[i]['lb'] = flb
            frs[i]['ub'] = fub
            frs[i]['idx_lb'] = fil
            frs[i]['idx_ub'] = fiu

            self.max_slice = np.maximum(self.max_slice, jnp.max(fiu + 1 - fil, axis=0))
        self.max_slice = tuple(np.astype(self.max_slice, int).tolist())

        print(f'- Forward reachable sets computed (took {(time.time() - t):.3f} sec.)')

        self.inputs = discrete_inputs
        self.idxs = np.arange(len(discrete_inputs))
        self.frs = frs

        print(f'Defining actions took {(time.time() - t_total):.3f} sec.')
        print('')
        return
