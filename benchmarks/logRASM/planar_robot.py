from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.dynamics.distributions import GaussianDistr, TriangularDistr
from benchmarks.dynamics import setmath
from core.rl.config import RLConfig


class PlanarRobot:
    '''
    Planar Robot benchmark from logRASM.
    3D state space (x, y, velocity) with steering control and obstacles to avoid while reaching a target set.
    '''

    def __init__(self, args):
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None

        self.n = 3
        self.p = 2
        self.state_variables = ['x', 'y', 'velocity']
        self.wrap = jnp.array([False, False, False], dtype=bool)
        self.plot_dimensions = [0, 1]
        self.lump = 1

        self.delta = 0.2

        self.uMin = np.array([-1.0, -1.0])
        self.uMax = np.array([1.0, 1.0])
        self.num_actions = [5, 5]

        noise_halfwidth = np.array([0.01, 0.01, 0.0])
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(noise_halfwidth ** 2)
            self.noise.set_partition_probs(num_cells=[10, 10, 1])
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(noise_halfwidth)
            self.noise.set_partition_probs(num_cells=[10, 10, 1])
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.partition = {}
        self.targets = {}
        self.partition['boundary'] = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([50, 50, 50])

        # Target set: x in [-1, -0.6], y in [0.6, 1.0], v in [-1, 1]
        self.goal = np.array([
            [[-1.0, 0.6, -1.0], [-0.6, 1.0, 1.0]],
        ], dtype=float)

        # 4 obstacle regions
        self.critical = np.array([
            [[-1.0, -1.0, -1.0], [-0.8, 0.0, 1.0]],
            [[-0.1, 0.8, -1.0], [1.0, 1.0, 1.0]],
            [[0.8, 0.0, -1.0], [1.0, 0.8, 1.0]],
            [[-0.4, -0.4, -1.0], [0.0, 0.1, 1.0]],
        ], dtype=float)

        self.x0 = np.array([
            [[0.4, -0.8, -0.1], [0.6, -0.6, 0.1]],
        ], dtype=float)

        self.rl_config = RLConfig(
            rl_algo="ppo",
            total_timesteps=2000000,
            RL_actions_per_state=9,
            inflation_rate=[(-2, 2), (-2, 2), (-2, 2)],
            per_step_cost=0.05,
            proximity_dims=[0, 1],
            proximity_penalty=0.5,
        )

    def step(self, state, action, noise):
        u = jnp.clip(action, self.uMin, self.uMax)
        v = state[2] + self.delta * 2.0 * u[0]
        x = state[0] + self.delta * v * jnp.cos(jnp.pi * u[1]) + noise[0]
        y = state[1] + self.delta * v * jnp.sin(jnp.pi * u[1]) + noise[1]
        v = v + noise[2]

        return jnp.array([x, y, v])

    @partial(jax.jit, static_argnums=(0,))
    def step_set(self, state_min, state_max, action_min, action_max):
        action_min = jnp.maximum(action_min, self.uMin)
        action_max = jnp.minimum(action_max, self.uMax)

        u0_min, u0_max = action_min[0], action_max[0]
        u1_min, u1_max = action_min[1], action_max[1]

        v_next_min = state_min[2] + self.delta * 2.0 * u0_min
        v_next_max = state_max[2] + self.delta * 2.0 * u0_max
        v_box = setmath.box(v_next_min, v_next_max)

        cos_min, cos_max = setmath.cos(jnp.pi * u1_min, jnp.pi * u1_max)
        sin_min, sin_max = setmath.sin(jnp.pi * u1_min, jnp.pi * u1_max)

        v_cos_min, v_cos_max = setmath.mult(v_box, [cos_min, cos_max])
        v_sin_min, v_sin_max = setmath.mult(v_box, [sin_min, sin_max])

        x_next_min = state_min[0] + self.delta * jnp.squeeze(v_cos_min)
        x_next_max = state_max[0] + self.delta * jnp.squeeze(v_cos_max)

        y_next_min = state_min[1] + self.delta * jnp.squeeze(v_sin_min)
        y_next_max = state_max[1] + self.delta * jnp.squeeze(v_sin_max)

        state_next_min = jnp.array([x_next_min, y_next_min, jnp.squeeze(v_box[0])])
        state_next_max = jnp.array([x_next_max, y_next_max, jnp.squeeze(v_box[1])])

        return state_next_min, state_next_max
