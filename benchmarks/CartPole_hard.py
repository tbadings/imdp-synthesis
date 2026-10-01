import jax.numpy as jnp
import numpy as np
from benchmarks.CartPole import CartPole
from core.rl.config import RLConfig


class CartPole_hard(CartPole):
    '''
    CartPole with an obstacle hanging above the middle of the track. The upright pole does not fit
    under the obstacle, so the pole must be tilted far enough while the cart passes underneath.
    '''

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u (force on the cart), both positive and negative
        self.uMin = [-10]
        self.uMax = [10]
        self.num_actions = [41]

        # The pole has to tilt beyond the obstacle's clearance angle (~0.14 rad), so the angle range
        # is wider than for CartPole.
        self.partition['boundary'] = np.array([[-2.4, -3.0, -0.5, -3.0],
                                               [2.4, 3.0, 0.5, 3.0]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([200, 200, 400, 200])

        # Obstacles hanging from the ceiling, as (x_min, x_max, bottom), where [x_min, x_max] is the
        # horizontal extent and bottom is the height of the lower edge above the pole's pivot.
        self.pole_length = 2 * self.length  # self.length is half the pole's length
        # The pole must tilt past ~8 deg to pass. Wider or lower obstacles need a longer or larger tilt,
        # which SAC does not learn reliably with this reward (e.g. 0.4 m wide at bottom 0.97 never succeeds).
        self.obstacles = [(-0.05, 0.05, 0.99)]

        # Goal: balance the pole upright on the right side of the track, past the obstacle
        self.goal = np.array([
            [[1, -1.5, -0.05, -1.0], [2.4, 1.5, 0.05, 1.0]]
        ], dtype=float)

        # Cart position and pole angle for which the pole hits an obstacle, covered by boxes
        self.critical = self.obstacle_boxes(num_bands=16)

        # Start upright on the left side of the obstacle
        self.x0 = np.array([-1, 0.0, 0, 0.0])

        # RL configuration: networks, SAC training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        # Penalties satisfy |penalty| > (per_step_cost + norm(distance_cost)) / (1 - gamma) = 15,
        # so that ending an episode on purpose never beats surviving.
        self.rl_config = RLConfig(
            rl_algo="sac",
            total_timesteps=5000000,
            goal_reward=20.0,
            unsafe_penalty=-20.0,
            out_of_bounds_penalty=-20.0,
            distance_cost=[0.1, 0.0, 0.0, 0.0],
            per_step_cost=0.05,
            inflation_rate=[(-7, 7), (-7, 7), (-7, 7), (-7, 7)],
            RL_actions_per_state=21,
        )

        return

    def obstacle_boxes(self, num_bands):
        '''
        Cover the states where the pole intersects an obstacle with boxes in (position, angle).

        A point at distance s along the pole is at (x + s*sin(angle), s*cos(angle)) relative to the
        pivot. For an obstacle (x_min, x_max, bottom), the part of the pole above the lower edge
        exists iff cos(angle) >= bottom / pole_length, and then spans the horizontal interval
        x + [min(o1, o2), max(o1, o2)] with o1 = pole_length*sin(angle), o2 = bottom*tan(angle).
        The pole thus hits the obstacle iff
            x_min - max(o1, o2) <= x <= x_max - min(o1, o2).
        Both offsets increase with the angle, so over an angle band [a_lo, a_hi] the union of these
        intervals is [x_min - max(o1, o2)(a_hi), x_max - min(o1, o2)(a_lo)], giving one box per band
        that covers the unsafe set exactly within that band.
        '''

        lower, upper = self.partition['boundary']
        boxes = []
        for x_min, x_max, bottom in self.obstacles:
            if bottom >= self.pole_length:
                continue
            clearance = np.arccos(bottom / self.pole_length)
            edges = np.linspace(-clearance, clearance, num_bands + 1)
            for a_lo, a_hi in zip(edges[:-1], edges[1:]):
                offset_hi = max(self.pole_length * np.sin(a_hi), bottom * np.tan(a_hi))
                offset_lo = min(self.pole_length * np.sin(a_lo), bottom * np.tan(a_lo))
                box_lb = lower.copy()
                box_ub = upper.copy()
                box_lb[[0, 2]] = [max(x_min - offset_hi, lower[0]), a_lo]
                box_ub[[0, 2]] = [min(x_max - offset_lo, upper[0]), a_hi]
                boxes.append([box_lb, box_ub])

        return np.array(boxes, dtype=float).reshape(-1, 2, self.n)
