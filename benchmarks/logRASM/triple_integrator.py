from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.dynamics.distributions import GaussianDistr, TriangularDistr
from benchmarks.dynamics import setmath
from core.rl.config import RLConfig


class TripleIntegrator:
    '''
    Triple integrator linear dynamics benchmark from logRASM.
    3D state space (absement, position, velocity) and 1D control input.
    '''

    def __init__(self, args):
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None

        self.n = 3
        self.p = 1
        self.state_variables = ['absement', 'position', 'velocity']
        self.wrap = jnp.array([False, False, False], dtype=bool)
        self.plot_dimensions = [0, 1]
        self.lump = 1

        self.A = np.array([
            [1.0, 0.045, 0.0],
            [0.0, 1.0, 0.045],
            [0.0, 0.0, 0.9],
        ])
        self.B = np.array([
            [0.35],
            [0.45],
            [0.5],
        ])
        self.W = np.diag([0.01, 0.01, 0.005])

        self.uMin = np.array([-1.0])
        self.uMax = np.array([1.0])
        self.num_actions = [11]

        noise_halfwidth = np.array([0.01, 0.01, 0.005])
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(noise_halfwidth ** 2)
            self.noise.set_partition_probs(num_cells=[10, 10, 10])
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(noise_halfwidth)
            self.noise.set_partition_probs(num_cells=[10, 10, 10])
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.partition = {}
        self.targets = {}
        self.partition['boundary'] = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([100, 100, 100])

        self.goal = np.array([
            [[-0.2, -0.2, -0.2], [0.2, 0.2, 0.2]],
        ], dtype=float)

        self.critical = np.array([
            [[-1.0, -1.0, -1.0], [-0.9, -0.9, 0.0]],
            [[0.9, 0.9, 0.0], [1.0, 1.0, 1.0]],
        ], dtype=float)

        self.x0 = np.array([
            [[-0.25, -0.25, -0.1], [-0.2, -0.2, 0.1]],
            [[0.2, 0.2, -0.1], [0.25, 0.25, 0.1]],
        ], dtype=float)

        self.rl_config = RLConfig(
            rl_algo="ppo",
            total_timesteps=500000,
            RL_actions_per_state=5,
            inflation_rate=[(-5, 5), (-5, 5), (-5, 5)],
            per_step_cost=0.05,
        )

    def step(self, state, action, noise):
        return jnp.dot(self.A, state) + jnp.dot(self.B, action) + noise

    @partial(jax.jit, static_argnums=(0,))
    def step_set(self, state_min, state_max, action_min, action_max):
        action_min = jnp.maximum(action_min, self.uMin)
        action_max = jnp.minimum(action_max, self.uMax)

        state_vertices = setmath.box2vertices(state_min, state_max)
        action_vertices = setmath.box2vertices(action_min, action_max)

        Ax = jnp.dot(self.A, state_vertices.T).T
        Bu = jnp.dot(self.B, action_vertices.T).T

        state_next_min = jnp.min(Ax, axis=0) + jnp.min(Bu, axis=0)
        state_next_max = jnp.max(Ax, axis=0) + jnp.max(Bu, axis=0)

        return state_next_min, state_next_max
