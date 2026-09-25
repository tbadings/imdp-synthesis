from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.dynamics.distributions import GaussianDistr, TriangularDistr
from benchmarks.dynamics import setmath
from core.rl.config import RLConfig


class CollisionAvoidance:
    '''
    Collision Avoidance benchmark from logRASM.
    2D state space (x, y) with non-linear repulsive potential field around two obstacles.
    '''

    def __init__(self, args):
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None

        self.n = 2
        self.p = 2
        self.state_variables = ['x', 'y']
        self.wrap = jnp.array([False, False], dtype=bool)
        self.plot_dimensions = [0, 1]
        self.lump = 1

        self.uMin = np.array([-1.0, -1.0])
        self.uMax = np.array([1.0, 1.0])
        self.num_actions = [7, 7]

        # Covariance / halfwidth of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.05, 0.05]) ** 2)
            self.noise.set_partition_probs(num_cells=[10, 10])
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.05, 0.05]))
            self.noise.set_partition_probs(num_cells=[10, 10])
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        self.partition = {}
        self.targets = {}
        self.partition['boundary'] = np.array([[-1.0, -1.0], [1.0, 1.0]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([50, 50])

        self.goal = np.array([
            [[-0.2, -0.2], [0.2, 0.2]],
        ], dtype=float)

        self.critical = np.array([
            [[-0.3, 0.7], [0.3, 1.0]],
            [[-0.3, -1.0], [0.3, -0.7]],
        ], dtype=float)

        self.x0 = np.array([
            [[-1.0, -0.6], [-0.9, 0.6]],
            [[0.9, -0.6], [1.0, 0.6]],
        ], dtype=float)

        self.rl_config = RLConfig(
            rl_algo="ppo",
            total_timesteps=100000,
            RL_actions_per_state=9,
            inflation_rate=[(-2, 2), (-2, 2)],
            per_step_cost=0.05,
            proximity_penalty=0.5,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step_base(self, state, action):
        u = 2.0 * jnp.clip(action, self.uMin, self.uMax)

        obstacle1 = jnp.array([0.0, 1.0])
        force1 = jnp.array([0.0, 1.0])
        dist1 = jnp.linalg.norm(obstacle1 - state)
        dist1 = jnp.clip(dist1 / 0.3, 0.0, 1.0)

        obstacle2 = jnp.array([0.0, -1.0])
        force2 = jnp.array([0.0, -1.0])
        dist2 = jnp.linalg.norm(obstacle2 - state)
        dist2 = jnp.clip(dist2 / 0.3, 0.0, 1.0)

        state_next = state + 0.2 * (dist2 * (u * dist1 + (1.0 - dist1) * force1) +
                                   (1.0 - dist2) * force2)
        return state_next

    def step(self, state, action, noise):
        return self.step_base(state, action) + noise

    @partial(jax.jit, static_argnums=(0,))
    def step_set(self, state_min, state_max, action_min, action_max):
        action_min = jnp.maximum(action_min, self.uMin)
        action_max = jnp.minimum(action_max, self.uMax)

        s_pts = setmath.box2vertices(state_min, state_max)
        s_center = (state_min + state_max) / 2.0
        s_all = jnp.vstack([s_pts, s_center[None, :]])

        a_pts = setmath.box2vertices(action_min, action_max)
        a_center = (action_min + action_max) / 2.0
        a_all = jnp.vstack([a_pts, a_center[None, :]])

        def step_for_state(s):
            return jax.vmap(lambda a: self.step_base(s, a))(a_all)

        evaluations = jax.vmap(step_for_state)(s_all)
        evaluations_flat = evaluations.reshape(-1, self.n)

        state_next_min = jnp.min(evaluations_flat, axis=0)
        state_next_max = jnp.max(evaluations_flat, axis=0)

        return state_next_min, state_next_max
