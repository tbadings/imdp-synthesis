from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.dynamics.distributions import GaussianDistr, TriangularDistr
from benchmarks.dynamics import setmath
from core.rl.config import RLConfig


class Drone4D:
    '''
    Drone 4D benchmark from logRASM.
    4D quadrotor state space (x1, v1, x2, v2) with crosswind and cubic damping.
    '''

    def __init__(self, args):
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None

        self.n = 4
        self.p = 2
        self.state_variables = ['x1', 'v1', 'x2', 'v2']
        self.wrap = jnp.array([False, False, False, False], dtype=bool)
        self.plot_dimensions = [0, 2]
        self.lump = 1

        self.tau = 0.5
        self.damping1 = 0.02
        self.damping2 = 0.01
        self.wind = -0.1

        self.uMin = np.array([-0.5, -0.5])
        self.uMax = np.array([0.5, 0.5])
        self.num_actions = [5, 5]

        noise_halfwidth = np.array([0.0, 0.01, 0.0, 0.01])
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(noise_halfwidth ** 2)
            self.noise.set_partition_probs(num_cells=[1, 10, 1, 10])
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(noise_halfwidth)
            self.noise.set_partition_probs(num_cells=[1, 10, 1, 10])
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.partition = {}
        self.targets = {}

        layout = 2

        if layout == 1:
            # DynAbs layout
            self.partition['boundary'] = np.array([[-1.0, -0.5, -1.0, -0.5], [0.0, 0.5, 1.0, 0.5]])
            self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
            self.partition['number_per_dim'] = np.array([20, 10, 20, 10])

            self.goal = np.array([
                [[-0.75, -0.5, 0.5, -0.5], [-0.5, 0.5, 0.75, 0.5]],
            ], dtype=float)

            self.critical = np.array([
                [[-1.0, -0.5, -0.25, -0.5], [-0.6, 0.5, 0.10, 0.5]],
            ], dtype=float)

            self.x0 = np.array([
                [[-0.9, -0.1, -0.9, -0.1], [-0.8, 0.1, -0.8, 0.1]],
            ], dtype=float)

        elif layout == 2:
            self.partition['boundary'] = np.array([[-0.5, -0.5, -0.5, -0.5], [0.5, 0.5, 0.5, 0.5]])
            self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
            self.partition['number_per_dim'] = np.array([50, 20, 50, 20])

            self.goal = np.array([
                [[0.3, -0.5, 0.3, -0.5], [0.5, 0.5, 0.5, 0.5]],
            ], dtype=float)

            self.critical = np.array([
                [[0.2, -0.5, -0.5, -0.5], [0.5, 0.5, -0.3, 0.5]],
                [[0.0, -0.5, -0.5, -0.5], [0.2, 0.5, -0.1, 0.5]],
                [[-0.5, -0.5, 0.4, -0.5], [0.0, 0.5, 0.5, 0.5]],
            ], dtype=float)

            self.x0 = np.array([
                [[-0.45, -0.1, -0.45, 0.25], [-0.35, 0.1, -0.35, 0.35]],
            ], dtype=float)

        else:
            # Standard layout
            self.partition['boundary'] = np.array([[-1.5, -1.5, -1.5, -1.5], [1.5, 1.5, 1.5, 1.5]])
            self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
            self.partition['number_per_dim'] = np.array([15, 15, 15, 15])

            self.goal = np.array([
                [[-0.2, -0.2, -0.2, -0.2], [0.2, 0.2, 0.2, 0.2]],
            ], dtype=float)

            self.critical = np.array([
                [[-1.5, -1.5, -1.5, -1.5], [-1.4, 0.0, -1.4, 0.0]],
                [[1.4, 0.0, 1.4, 0.0], [1.5, 1.5, 1.5, 1.5]],
            ], dtype=float)

            self.x0 = np.array([
                [[-0.25, -0.1, -0.25, -0.1], [-0.20, 0.1, -0.20, 0.1]],
                [[0.20, -0.1, 0.20, -0.1], [0.25, 0.1, 0.25, 0.1]],
            ], dtype=float)

        self.rl_config = RLConfig(
            rl_algo="ppo",
            total_timesteps=1000000,
            RL_actions_per_state=9,
            inflation_rate=[(-4, 4), (-4, 4), (-4, 4), (-4, 4)],
            proximity_dims=[0, 2],
            proximity_penalty=0.5,
            per_step_cost=0.05,
        )

    def step(self, state, action, noise):
        u = jnp.clip(action, self.uMin, self.uMax)
        x1, v1, x2, v2 = state[0], state[1], state[2], state[3]

        x2_next = x2 + self.tau * v2 + (self.tau ** 2 / 2.0) * u[1] + noise[2]
        v2_next = v2 + self.tau * (-self.damping2 * (v2 ** 3) + u[1] + self.wind * jnp.sin(jnp.pi * x1)) + noise[3]
        x1_next = x1 + self.tau * v1 + (self.tau ** 2 / 2.0) * u[0] + noise[0]
        v1_next = v1 + self.tau * (-self.damping1 * (v1 ** 3) + u[0]) + noise[1]

        return jnp.array([x1_next, v1_next, x2_next, v2_next])

    @partial(jax.jit, static_argnums=(0,))
    def step_set(self, state_min, state_max, action_min, action_max):
        action_min = jnp.maximum(action_min, self.uMin)
        action_max = jnp.minimum(action_max, self.uMax)

        x1_min, v1_min, x2_min, v2_min = state_min[0], state_min[1], state_min[2], state_min[3]
        x1_max, v1_max, x2_max, v2_max = state_max[0], state_max[1], state_max[2], state_max[3]

        # In the bounded velocity regime (|v| <= 1.5), v -> v - tau * damping * v^3 is strictly increasing
        v1_term_min = v1_min - self.tau * self.damping1 * (v1_min ** 3)
        v1_term_max = v1_max - self.tau * self.damping1 * (v1_max ** 3)

        v2_term_min = v2_min - self.tau * self.damping2 * (v2_min ** 3)
        v2_term_max = v2_max - self.tau * self.damping2 * (v2_max ** 3)

        sin_min, sin_max = setmath.sin(jnp.pi * x1_min, jnp.pi * x1_max)
        # self.wind is negative (-0.1), so multiplication flips min and max
        wind_min = self.tau * self.wind * sin_max
        wind_max = self.tau * self.wind * sin_min

        x1_next_min = x1_min + self.tau * v1_min + (self.tau ** 2 / 2.0) * action_min[0]
        x1_next_max = x1_max + self.tau * v1_max + (self.tau ** 2 / 2.0) * action_max[0]

        v1_next_min = v1_term_min + self.tau * action_min[0]
        v1_next_max = v1_term_max + self.tau * action_max[0]

        x2_next_min = x2_min + self.tau * v2_min + (self.tau ** 2 / 2.0) * action_min[1]
        x2_next_max = x2_max + self.tau * v2_max + (self.tau ** 2 / 2.0) * action_max[1]

        v2_next_min = v2_term_min + self.tau * action_min[1] + wind_min
        v2_next_max = v2_term_max + self.tau * action_max[1] + wind_max

        state_next_min = jnp.array([x1_next_min, v1_next_min, x2_next_min, v2_next_min])
        state_next_max = jnp.array([x1_next_max, v1_next_max, x2_next_max, v2_next_max])

        return state_next_min, state_next_max
