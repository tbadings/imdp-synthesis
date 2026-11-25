'''
Quadcopter benchmarks: a double integrator per position dimension, in 2D (Quadcopter2D) or 3D (Quadcopter3D), optionally with a
battery state (Quadcopter2D_battery, Quadcopter3D_battery) or as two independent quadcopters (Quadcopter2D_2agent).
'''

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
import jax.numpy as jnp
import numpy as np
import scipy

from benchmarks.base import Benchmark
from benchmarks.dynamics import setmath
from core.abstraction.partition import EPS
from core.box import Box
from core.plotting.utils import save_fig
from core.rl.config import RLConfig


class QuadcopterDynamics(Benchmark):
    '''
    Quadcopter moving in dim = 2 or 3 dimensions: a double integrator per dimension, with state (position, velocity) per
    dimension and an acceleration input per dimension. Velocities are damped quadratically (--damping) and saturate
    at [v_min, v_max].
    '''

    v_min = -2.0
    v_max = 2.0

    def __init__(self, args, dim=2):
        self._set_motion(args, dim)
        if dim == 2:
            self.set_noise(args,
                           gaussian=dict(variance=np.array([0.15, 0, 0.15, 0])**2, cells=[10, 1, 10, 1]),
                           triangular=dict(halfwidth=[0.15, 0, 0.15, 0], cells=[10, 1, 10, 1]))
        else:
            self.set_noise(args,
                           gaussian=dict(variance=np.array([0.05, 0, 0.05, 0, 0.05, 0])**2, cells=[5, 1, 5, 1, 5, 1]),
                           triangular=dict(halfwidth=[0.1, 0, 0.1, 0, 0.1, 0], cells=[10, 1, 10, 1, 10, 1]))

    def _set_motion(self, args, dim, agents=1, battery=False):
        '''
        Set the state variables and the matrices A, B of the motion: `agents` quadcopters in `dim` dimensions each, plus a
        last, battery state (whose row of A is the identity) if battery.
        '''
        if dim not in (2, 3):
            raise ValueError(f'Quadcopter dynamics only support dim in [2, 3], got {dim}')

        self.dim = dim
        self.tau = 1.0
        self.damping = args.damping

        variables =[f'{axis}_{quantity}' for axis in 'xyz'[:dim] for quantity in ('pos', 'vel')]
        if agents > 1:
            variables = [f'quadcopter_{agent}_{variable}' for agent in range(1, agents + 1) for variable in variables]

        Ablock = np.array([[1, self.tau],
                           [0, 1]])
        Bblock = np.array([[self.tau**2/2],
                           [self.tau]])
        self.A = scipy.linalg.block_diag(*[Ablock] * (dim * agents))
        self.B = scipy.linalg.block_diag(*[Bblock] * (dim * agents))

        if battery:
            variables = variables + ['battery']
            self.A = scipy.linalg.block_diag(self.A, 1)
            self.B = np.vstack([self.B, np.zeros((1, self.B.shape[1]))])
        self.state_variables = variables

    def step(self, state, action, noise):
        state_next = jnp.dot(self.A, state) + jnp.dot(self.B, action) + noise
        state_next = state_next.at[1::2].add(-self.damping * state[1::2] * jnp.abs(state[1::2]))
        state_next = state_next.at[1::2].set(self.saturate(state_next[1::2], self.v_min, self.v_max))

        return state_next

    def successor_bounds(self, state_min, state_max, action_min, action_max):
        '''Bounds on the successors of the motion (the battery state, if any, stays constant).'''

        # Get vertices of the state and action boxes
        state_vertices = setmath.box2vertices(state_min, state_max)
        action_vertices = setmath.box2vertices(action_min, action_max)

        # Propagate dynamics for all vertices
        Ax = jnp.dot(self.A, state_vertices.T).T  # Shape (2^n, n)
        Ax = Ax.at[:, 1::2].add(-self.damping * state_vertices[:, 1::2] * jnp.abs(state_vertices[:, 1::2]))
        Bu = jnp.dot(self.B, action_vertices.T).T  # Shape (2^p, n)

        # Combine min/max to get the reachable set
        state_next_min = jnp.min(Ax, axis=0) + jnp.min(Bu, axis=0)
        state_next_max = jnp.max(Ax, axis=0) + jnp.max(Bu, axis=0)

        state_next_min = state_next_min.at[1::2].set(self.saturate(state_next_min[1::2], self.v_min, self.v_max))
        state_next_max = state_next_max.at[1::2].set(self.saturate(state_next_max[1::2], self.v_min, self.v_max))

        return state_next_min, state_next_max

    def plot_simulation(self, args, stamp, partition, traces):
        '''A quadcopter moving in 3D: plot the traces in 3D, with Matplotlib and PyVista.'''
        if self.dim != 3:
            return
        from core.plotting.quadcopter3D import plot_quadcopter3D_pyvista
        from core.plotting.traces import plot_traces_3D

        print('Plot Quadcopter3D traces in 3D...')
        plot_traces_3D(args, stamp, [0, 2, 4], partition, self, traces, num_traces=100, filename="traces_3D")
        plot_quadcopter3D_pyvista(args, stamp, [0, 2, 4], partition, self, traces, num_traces=10)


class QuadcopterDynamics_2agent(QuadcopterDynamics):
    '''Two independent 2D quadcopters: quadcopter 1 in state dimensions 0-3 and input dimensions 0-1, quadcopter 2 in 4-7 and 2-3.'''

    def __init__(self, args):
        self._set_motion(args, dim=2, agents=2)
        # Gaussian noise only acts on the x position of quadcopter 1 and the y position of quadcopter 2; the triangular noise is
        # that of a single 2D quadcopter, for each quadcopter
        self.set_noise(args,
                       gaussian=dict(variance=np.array([0.05, 0, 0, 0, 0, 0, 0.05, 0])**2,
                                     cells=[5, 1, 1, 1, 1, 1, 5, 1]),
                       triangular=dict(halfwidth=np.tile([0.15, 0, 0.15, 0], 2), cells=[10, 1, 10, 1] * 2))


class QuadcopterDynamics_battery(QuadcopterDynamics):
    '''
    QuadcopterDynamics with a battery state (the last dimension): in the charging station, the battery charges by 10 per
    step up to max_charge; elsewhere it drains by 5 per step. Benchmarks define charging_station (a list of one Box).
    '''

    max_charge = 100

    def __init__(self, args, dim=2):
        self._set_motion(args, dim, battery=True)
        if dim == 2:
            self.set_noise(args,
                           gaussian=dict(variance=np.array([0.15, 0, 0.15, 0, 0])**2, cells=[10, 1, 10, 1, 1]),
                           triangular=dict(halfwidth=[0.1, 0, 0.1, 0, 0], cells=[10, 1, 10, 1, 1]))
        else:
            self.set_noise(args,
                           gaussian=dict(variance=np.array([0.05, 0, 0, 0, 0.05, 0, 0])**2,
                                         cells=[5, 1, 1, 1, 5, 1, 1]),
                           triangular=dict(halfwidth=[0.1, 0, 0.1, 0, 0.1, 0, 0], cells=[10, 1, 10, 1, 10, 1, 1]))

    def step(self, state, action, noise):
        state_next = super().step(state, action, noise)
        battery_idx = self.n - 1

        station = self.charging_station[0]
        in_station = jnp.all((state >= station.lower) & (state <= station.upper))
        new_battery = jnp.where(
            in_station,
            self.saturate(state_next[battery_idx] + 10, high=self.max_charge),
            state_next[battery_idx] - 5
        )
        return state_next.at[battery_idx].set(new_battery)

    def successor_bounds(self, state_min, state_max, action_min, action_max):
        state_next_min, state_next_max = super().successor_bounds(state_min, state_max, action_min, action_max)

        # Battery charging calculation (+10 when charging, capped at max_charge; -5 otherwise)
        cs_min = self.charging_station[0].lower
        cs_max = self.charging_station[0].upper

        # Same tolerances as the goal (containment) and critical (overlap) regions of the partition: a cell
        # that only touches the station does not intersect it, and cell bounds that round past a station face
        # (e.g. the top velocity cell's upper bound, 2.0000002 for v_max = 2) still count as inside
        entirely_inside = jnp.all((state_min >= cs_min - EPS / 2) & (state_max <= cs_max + EPS / 2))
        intersects = jnp.all((state_max >= cs_min + EPS) & (state_min <= cs_max - EPS))

        # entirely_inside: every state charges -> both bounds +10
        # partial overlap: some states charge, some drain -> min -5, max +10
        # no overlap: every state drains -> both bounds -5
        delta_min = jnp.where(entirely_inside, 10.0, -5.0)
        delta_max = jnp.where(intersects, 10.0, -5.0)

        battery_idx = self.n - 1
        new_min = self.saturate(state_next_min[battery_idx] + delta_min, high=self.max_charge)
        new_max = self.saturate(state_next_max[battery_idx] + delta_max, high=self.max_charge)
        state_next_min = state_next_min.at[battery_idx].set(new_min)
        state_next_max = state_next_max.at[battery_idx].set(new_max)

        return state_next_min, state_next_max


class Quadcopter2D(QuadcopterDynamics):
    '''
    Quadcopter benchmark, with a 4D state space and a 2D control input space.
    '''

    def __init__(self, args):
        QuadcopterDynamics.__init__(self, args, dim=2)

        self.plot_dimensions = [0, 2]

        v_min, v_max = self.v_min, self.v_max
        self.state_space = Box([-20.0, v_min, -20.0, v_min], [20.0, v_max, 20.0, v_max], names=self.state_variables)
        self.state_grid = [80, 10, 80, 10]
        self.input_space = Box([-1, -1], [1, 1])
        self.input_grid = [5, 5]

        region = self.state_space.restrict
        self.goal = [region(x_pos=(12.0, 18.0), y_pos=(12.0, 18.0))]
        self.critical = [
            region(x_pos=(-20.0, 8.0), y_pos=(-7.0, -4.0)),     # Lower wall with right-side opening
            region(x_pos=(-8.0, 20.0), y_pos=(4.0, 7.0)),       # Upper wall with left-side opening
            region(x_pos=(-4.0, 4.0), y_pos=(-14.0, -10.0)),    # Lower lane obstacle
            region(x_pos=(-4.0, 4.0), y_pos=(10.0, 14.0)),      # Upper lane obstacle
        ]

        self.x0 = np.array([-15.0, 0.01, -15.0, 0.01])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="ppo",
            # TODO: Long training is still needed here; can we reduce that?
            total_timesteps=3000000,
            RL_actions_per_state=9,
            inflation_rate=[(-4, 4), (-2, 2), (-4, 4), (-2, 2)],
            proximity_dims = [0, 2],
            proximity_penalty=0.5,
            per_step_cost=0.05,
        )


class Quadcopter2D_battery(QuadcopterDynamics_battery):
    '''
    Quadcopter benchmark, with a 4D state space and a 2D control input space, plus a battery state.
    '''

    def __init__(self, args):
        QuadcopterDynamics_battery.__init__(self, args, dim=2)

        self.plot_dimensions = [0, 2]

        v_min, v_max = self.v_min, self.v_max
        self.state_space = Box([-10, v_min, -10, v_min, 0], [10, v_max, 10, v_max, self.max_charge],
                               names=self.state_variables)
        self.state_grid = [40, 10, 40, 10, 20]
        self.input_space = Box([-1, -1], [1, 1])
        self.input_grid = [5, 5]

        region = self.state_space.restrict
        self.goal = [region(x_pos=(6, 10), y_pos=(6, 10), battery=(50, self.max_charge))]
        self.critical = [
            # Obstacle in the center
            region(x_pos=(-5, 5), y_pos=(-1, 1)),
            region(x_pos=(-1, 1), y_pos=(-5, 5)),
        ]
        self.charging_station = [region(x_pos=(-9, -6), y_pos=(-5, 5))]

        self.x0 = np.array([0.01, 0.01, -9, 0.01, 50])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="ppo",
            total_timesteps=1000000,
            proximity_dims = [0, 2],
            proximity_penalty=0.5,
            per_step_cost=0.05,
            inflation_rate=[(-4, 4), (-2, 2), (-4, 4), (-2, 2), (-2, 2)],
            RL_actions_per_state=25,
        )


# Obstacles of Quadcopter3D and Quadcopter3D_battery, as position bounds
QUADCOPTER3D_OBSTACLES = [
    # Hole 1
    dict(x_pos=(-11, -5), y_pos=(-1, 9), z_pos=(-7, -5)),
    dict(x_pos=(-11, -5), y_pos=(5, 9), z_pos=(-5, 5)),
    dict(x_pos=(-11, -5), y_pos=(-1, 3), z_pos=(-5, 3)),
    # Hole 2
    dict(x_pos=(-1, 3), y_pos=(1, 9), z_pos=(-7, -1)),
    dict(x_pos=(-1, 3), y_pos=(1, 9), z_pos=(3, 5)),
    dict(x_pos=(-1, 3), y_pos=(1, 3), z_pos=(-1, 3)),
    dict(x_pos=(-1, 3), y_pos=(7, 9), z_pos=(-1, 3)),
    # Tower
    dict(x_pos=(-1, 3), y_pos=(-3, 1), z_pos=(-7, 7)),
    # Wall between routes
    dict(x_pos=(3, 9), y_pos=(-3, 1), z_pos=(-7, -1)),
    # Long route obstacles
    dict(x_pos=(-1, 3), y_pos=(-9, -3), z_pos=(-7, -5)),
    # Overhanging
    dict(x_pos=(-1, 3), y_pos=(-9, -3), z_pos=(3, 7)),
    # Small last obstacle
    dict(x_pos=(11, 15), y_pos=(-9, -5), z_pos=(-7, -5)),
    # Obstacle next to goal
    dict(x_pos=(9, 15), y_pos=(5, 9), z_pos=(-7, 1)),
]


class Quadcopter3D(QuadcopterDynamics):
    '''
    Quadcopter benchmark, with a 6D state space and a 3D control input space.
    '''

    def __init__(self, args):
        QuadcopterDynamics.__init__(self, args, dim=3)

        self.plot_dimensions = [0, 2]

        v_min, v_max = self.v_min, self.v_max
        self.state_space = Box([-17, v_min, -9, v_min, -7, v_min], [17, v_max, 9, v_max, 7, v_max],
                               names=self.state_variables)
        self.state_grid = [68, 8, 36, 8, 28, 8]
        self.input_space = Box([-1, -1, -1], [1, 1, 1])
        self.input_grid = [5, 5, 5]

        region = self.state_space.restrict
        self.goal = [region(x_pos=(11, 15), y_pos=(1, 5), z_pos=(-7, -3))]
        self.critical = [region(**obstacle) for obstacle in QUADCOPTER3D_OBSTACLES]

        self.x0 = np.array([-14.5, 0.01, 6, 0.01, 2, 0.01])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="sac",
            total_timesteps=10000000,
            RL_actions_per_state=27,
            proximity_dims=[0, 2, 4],
            proximity_penalty=0.5,
            per_step_cost=0.05,
            inflation_rate=[(-3, 3), (-2, 2), (-3, 3), (-2, 2), (-3, 3), (-2, 2)],
        )


class Quadcopter3D_small(QuadcopterDynamics):
    '''
    Reduced Quadcopter3D configuration for faster debugging, with a 6D state space and a 3D control input space.
    '''

    def __init__(self, args):
        QuadcopterDynamics.__init__(self, args, dim=3)

        self.plot_dimensions = [0, 2]

        v_min, v_max = self.v_min, self.v_max
        self.state_space = Box([-7, v_min, -7, v_min, -7, v_min], [7, v_max, 7, v_max, 7, v_max],
                               names=self.state_variables)
        self.state_grid = [28, 10, 28, 10, 28, 10]
        self.input_space = Box([-1, -1, -1], [1, 1, 1])
        self.input_grid = [5, 5, 5]

        region = self.state_space.restrict
        self.goal = [region(x_pos=(3, 7), y_pos=(3, 7))]
        self.critical = [
            region(x_pos=(-7, -1), y_pos=(1, 3)),
            region(x_pos=(3, 7), y_pos=(-7, -3)),
        ]

        self.x0 = np.array([-5.5, 0.01, -5.5, 0.01, 0.01, 0.01])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            pi_arch=[32, 32],
            vf_arch=[32, 32],
            total_timesteps=100000,
            eval_episodes=1000,
            goal_reward=5,
            unsafe_penalty=-5,
            out_of_bounds_penalty=-5,
            per_step_cost=0.1,
            distance_cost=0.0,
            RL_actions_per_state=27,
            inflation_rate=[(-2, 2), (-1, 1), (-2, 2), (-1, 1), (-2, 2), (-1, 1)],
        )


class Quadcopter3D_battery(QuadcopterDynamics_battery):
    '''
    Quadcopter benchmark with 6D motion, one battery state, and 3D control.
    '''

    def __init__(self, args):
        QuadcopterDynamics_battery.__init__(self, args, dim=3)

        self.plot_dimensions = [0, 2]

        # Same layout as Quadcopter3D (bounds, obstacles, goal, and initial state), plus the battery dimension
        v_min, v_max = self.v_min, self.v_max
        self.state_space = Box([-17, v_min, -9, v_min, -7, v_min, 0], [17, v_max, 9, v_max, 7, v_max, self.max_charge],
                               names=self.state_variables)
        self.state_grid = [68, 8, 36, 8, 28, 8, 20]
        self.input_space = Box([-1, -1, -1], [1, 1, 1])
        self.input_grid = [5, 5, 5]

        region = self.state_space.restrict
        self.goal = [region(x_pos=(11, 15), y_pos=(1, 5), z_pos=(-7, -3), battery=(20, self.max_charge))]
        self.critical = [region(**obstacle) for obstacle in QUADCOPTER3D_OBSTACLES]

        # Charging pad in the minimal (x, y, z) corner of the state space
        self.charging_station = [region(x_pos=(-15, -7), y_pos=(-9, -4), z_pos=(-7, 0))]

        self.x0 = np.array([-14.5, 0.01, 6, 0.01, 2, 0.01, 50])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="sac",
            total_timesteps=25000000,
            RL_actions_per_state=27,
            proximity_dims=[0, 2, 4],
            proximity_penalty=0.5,
            per_step_cost=0.05,
            unsafe_penalty=-10,
            out_of_bounds_penalty=-10,
            inflation_rate=[(-3, 3), (-2, 2), (-3, 3), (-2, 2), (-3, 3), (-2, 2), (-2, 2)],
        )


class Quadcopter2D_2agent(QuadcopterDynamics_2agent):
    '''
    Two independent Quadcopter2D agents, with an 8D state space and a 4D control input space.
    '''

    # Position dimensions (x, y) of each quadcopter, and the colors to plot the quadcopters in
    POSITION_DIMS = [(0, 2), (4, 6)]
    QUADCOPTER_COLORS = ['tab:blue', 'tab:orange']

    def __init__(self, args):
        QuadcopterDynamics_2agent.__init__(self, args)

        self.plot_dimensions = [0, 2]

        v_min, v_max = self.v_min, self.v_max
        self.state_space = Box([-2.5, v_min, -2.5, v_min, -2.5, v_min, -2.5, v_min],
                               [2.5, v_max, 2.5, v_max, 2.5, v_max, 2.5, v_max], names=self.state_variables)
        self.state_grid = [20, 8, 20, 8, 20, 8, 20, 8]
        self.input_space = Box([-1, -1, -1, -1], [1, 1, 1, 1])
        self.input_grid = [5, 5, 5, 5]

        # Goal: the quadcopters swap sides
        region = self.state_space.restrict
        self.goal = [
            region(quadcopter_1_x_pos=(-2.5, -1.25), quadcopter_1_y_pos=(-1, 1),
                   quadcopter_2_x_pos=(1.25, 2.5), quadcopter_2_y_pos=(-1, 1)),
            region(quadcopter_1_x_pos=(1.25, 2.5), quadcopter_1_y_pos=(-1, 1),
                   quadcopter_2_x_pos=(-2.5, -1.25), quadcopter_2_y_pos=(-1, 1)),
        ]

        # Obstacles in the (x, y) position plane; both quadcopters must avoid them
        self.obstacles = [
            Box([-0.5, -0.5], [0.5, 0.5]),
            Box([-2.5, 1], [-0.5, 1.5]),
        ]

        # Critical: either quadcopter in an obstacle, or both quadcopters in the same position cell (a collision)
        self.critical = self.obstacle_boxes(self.obstacles) + self.collision_boxes()

        self.x0 = np.array([-1.5, 0, 2.25, -0.01, 1.5, 0, -2.25, 0.01])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="ppo",
            total_timesteps=5000000,
            RL_actions_per_state=3**4,
            inflation_rate=[(-3, 3), (-1, 1), (-3, 3), (-1, 1), (-3, 3), (-1, 1), (-3, 3), (-1, 1)],
                        # [(-4, 4), (-2, 2), (-4, 4), (-2, 2), (-4, 4), (-2, 2), (-4, 4), (-2, 2)],
            proximity_dims = [0, 2, 4, 6],
            goal_reward=50.0,
            unsafe_penalty=-50.0,
            out_of_bounds_penalty=-50.0,
            distance_cost=[0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0],
            per_step_cost=0.01,
            proximity_penalty=0.1,
            # Many rollouts, all starting inside the cell of x0 (no margin): in 8D, a 10% margin per side puts
            # three quarters of the starts outside that cell. 300 rollouts give V(s0) ~ 0.999 for SAC seeds 0-4,
            # whereas 100 rollouts with the margin gave 0.53 (seed 0).
            eval_episodes=300,
            reset_margin=0.0,
            pi_arch=[256, 256],
            vf_arch=[256, 256],
        )

    def obstacle_boxes(self, obstacles):
        '''
        Lift obstacles in the position plane to boxes in the state space, one per obstacle and quadcopter.

        The box of a quadcopter restricts that quadcopter's position (x, y) to the obstacle and leaves all other dimensions
        free, so it covers the states where that quadcopter is in the obstacle, wherever the other quadcopter is.

        Args:
            obstacles: list of Boxes in the (x, y) position plane
        '''

        return [self.state_space.restrict({ix: (obstacle.lower[0], obstacle.upper[0]),
                                           iy: (obstacle.lower[1], obstacle.upper[1])})
                for obstacle in obstacles for ix, iy in self.POSITION_DIMS]

    def collision_boxes(self):
        '''
        Cover the states where both quadcopters are in the same position cell of the partition grid with boxes.

        There is one box per position cell (an x cell and a y cell): it restricts the x positions of both
        quadcopters (dimensions 0 and 4) to the x cell and their y positions (dimensions 2 and 6) to the y cell,
        and leaves the velocities free. The partition marks a cell as critical iff both quadcopters are in the
        same position cell, as boxes that only touch a cell do not count as overlapping it.
        '''

        lower, upper, number = self.state_space.lower, self.state_space.upper, self.state_grid
        for d1, d2 in [(0, 4), (2, 6)]:
            assert lower[d1] == lower[d2] and upper[d1] == upper[d2] and number[d1] == number[d2], \
                f"Both quadcopters need the same position grid in dimensions {d1} and {d2}"

        x_edges = np.linspace(lower[0], upper[0], number[0] + 1)
        y_edges = np.linspace(lower[2], upper[2], number[2] + 1)

        return [self.state_space.restrict({0: (x_lo, x_hi), 4: (x_lo, x_hi), 2: (y_lo, y_hi), 6: (y_lo, y_hi)})
                for x_lo, x_hi in zip(x_edges[:-1], x_edges[1:])
                for y_lo, y_hi in zip(y_edges[:-1], y_edges[1:])]

    def _plot_position_plane(self, ax):
        '''
        Draw the position plane shared by both quadcopters: the position cells of the partition, each quadcopter's goal, and
        the obstacles.
        '''

        lower, upper, number = self.state_space.lower, self.state_space.upper, self.state_grid

        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[2], upper[2])
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xticks(np.linspace(lower[0], upper[0], number[0] + 1), minor=True)
        ax.set_yticks(np.linspace(lower[2], upper[2], number[2] + 1), minor=True)
        ax.grid(which='minor', color='lightgray', lw=0.5)

        for goal in self.goal:
            for agent, ((ix, iy), color) in enumerate(zip(self.POSITION_DIMS, self.QUADCOPTER_COLORS)):
                ax.add_patch(Rectangle((goal.lower[ix], goal.lower[iy]), goal.width[ix], goal.width[iy],
                                       facecolor=color, edgecolor=color, alpha=0.2, hatch='//', lw=0,
                                       label=f'Goal quadcopter {agent + 1}'))

        for obstacle in self.obstacles:
            ax.add_patch(Rectangle(obstacle.lower, *obstacle.width, facecolor='dimgray', alpha=0.6, lw=0,
                                   label='Obstacle'))

    def plot_simulation(self, args, stamp, partition, traces):
        print('Plot Quadcopter2D_2agent trace and gif...')
        trajectory = np.array(traces[0]['x'])
        self.plot_trace(trajectory, filename=args.output_dir / f'quadcopter2D_2agent_trace_{stamp}')
        self.plot_trajectory_gif(trajectory, filename=str(args.output_dir / f'quadcopter2D_2agent_{stamp}.gif'))

    def plot_trace(self, trajectory, filename):
        '''
        Plots a trajectory of both quadcopters in the position plane, with the time steps of the states.

        Args:
            trajectory: np.ndarray of shape (T, 8), the states of the trajectory
            filename: Output path without extension (stored as pdf and png)
        '''

        trajectory = np.asarray(trajectory, dtype=float)

        fig, ax = plt.subplots(figsize=(6, 6))
        self._plot_position_plane(ax)

        for agent, ((ix, iy), color) in enumerate(zip(self.POSITION_DIMS, self.QUADCOPTER_COLORS)):
            ax.plot(trajectory[:, ix], trajectory[:, iy], '-o', color=color, lw=1.5, ms=4, label=f'Quadcopter {agent + 1}')
            ax.plot(trajectory[0, ix], trajectory[0, iy], 's', color=color, ms=8)

            # Consecutive states that barely move (e.g., hovering in the goal) share one label 'first-last'
            positions = trajectory[:, [ix, iy]]
            first = 0
            for k in range(1, len(positions) + 1):
                if k == len(positions) or np.linalg.norm(positions[k] - positions[first]) > 0.15:
                    label = str(first) if k - 1 == first else f'{first}–{k - 1}'
                    ax.annotate(label, positions[first], xytext=(4, 4), textcoords='offset points', fontsize=7,
                                color=color)
                    first = k

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        save_fig(fig, filename)

    def plot_trajectory_gif(self, trajectory, filename="quadcopter2D_2agent_trajectory.gif", substeps=10):
        '''
        Plots a trajectory of both quadcopters as an animation and stores it as a gif.

        Positions are interpolated linearly between time steps, and the trails mark the states at the time steps.
        The position cell of each quadcopter at the last time step is shaded, in red if both quadcopters are in the same cell
        (a collision).

        Args:
            trajectory: np.ndarray of shape (T, 8), the states of the trajectory
            filename: Output filename for the gif
            substeps: Number of frames per time step
        '''

        trajectory = np.asarray(trajectory, dtype=float)
        lower, upper = self.state_space.lower, self.state_space.upper
        width = (upper - lower) / np.asarray(self.state_grid)

        fig, ax = plt.subplots(figsize=(6, 6))
        self._plot_position_plane(ax)

        trails, markers, cells = [], [], []
        for agent, ((ix, iy), color) in enumerate(zip(self.POSITION_DIMS, self.QUADCOPTER_COLORS)):
            trails.append(ax.plot([], [], '-o', color=color, lw=1.5, ms=4, label=f'Quadcopter {agent + 1}')[0])
            markers.append(ax.plot([], [], 'o', color=color, ms=10)[0])
            cells.append(ax.add_patch(Rectangle((0, 0), width[ix], width[iy], facecolor=color, alpha=0.4)))
        time_text = ax.set_title('step 0')  # Not empty, so that tight_layout leaves room for it
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        fig.tight_layout()

        # Frames between the time steps, plus frames that hold the final state
        num_frames = (len(trajectory) - 1) * substeps + 1
        hold_frames = 2 * substeps

        def animate(i):
            k, sub = divmod(min(i, num_frames - 1), substeps)
            frac = sub / substeps
            state = (1 - frac) * trajectory[k] + frac * trajectory[min(k + 1, len(trajectory) - 1)]

            # Grid cell of the state at the last time step; a collision if both quadcopters share a position cell
            cell_idx = np.floor((trajectory[k] - lower) / width)
            collision = np.array_equal(*(cell_idx[list(dims)] for dims in self.POSITION_DIMS))

            for (ix, iy), color, trail, marker, cell in zip(self.POSITION_DIMS, self.QUADCOPTER_COLORS, trails, markers, cells):
                path = np.vstack([trajectory[:k + 1, [ix, iy]], state[[ix, iy]]])
                trail.set_data(path[:, 0], path[:, 1])
                trail.set_markevery(list(range(k + 1)))
                marker.set_data([state[ix]], [state[iy]])
                cell.set_xy(lower[[ix, iy]] + cell_idx[[ix, iy]] * width[[ix, iy]])
                cell.set_facecolor('crimson' if collision else color)
            time_text.set_text(f'step {k}' + (' (collision)' if collision else ''))
            return *trails, *markers, *cells, time_text

        ani = animation.FuncAnimation(
            fig, animate, frames=num_frames + hold_frames, interval=50, blit=True
        )

        ani.save(filename, writer='pillow', dpi=100)
        plt.close(fig)
