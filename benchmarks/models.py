'''
This file contains the dynamics models used in the benchmarks.
'''

from functools import partial
import logging

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from benchmarks.dynamics.distributions import GaussianDistr, TriangularDistr
from benchmarks.dynamics import setmath

logger = logging.getLogger(__name__)


def wrap_theta(theta):
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

class DubinsDynamics3D:
    def __init__(self, args):
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None

        # Discretization step size
        self.tau = 1

        self.n = 3
        self.p = 2
        self.state_variables = ['x', 'y', 'angle']
        self.wrap = jnp.array([False, False, True], dtype=bool)

        self.alpha_min = 0.85
        self.alpha_max = 0.85
        self.alpha = 0.85

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0, 0, 0.1])**2) # From stdev to covariance
            self.noise.set_partition_probs(num_cells=[1, 1, 10])
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0, 0, 0.2])) # Halfwidth
            self.noise.set_partition_probs(num_cells=[1, 1, 10])
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

    def step(self, state, action, noise):
        x, y, theta = state[0], state[1], state[2]
        u1, u2 = action[0], action[1]
        x_next = x + self.tau * u2 * jnp.cos(theta)
        y_next = y + self.tau * u2 * jnp.sin(theta)
        theta_next = wrap_theta(theta + self.tau * self.alpha * u1 + noise[2])

        state_next = jnp.array([x_next, y_next, theta_next])

        return state_next

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        # Convert to boxes
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [x_min, y_min, theta_min] = state_min
        [x_max, y_max, theta_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        [u1_min, u2_min] = jnp.maximum(action_min, self.uMin)
        [u1_max, u2_max] = jnp.minimum(action_max, self.uMax)

        x_next = jnp.array([x_min, x_max]) + self.tau * jnp.concat(setmath.mult([u2_min, u2_max], setmath.cos(theta_min, theta_max)))
        y_next = jnp.array([y_min, y_max]) + self.tau * jnp.concat(setmath.mult([u2_min, u2_max], setmath.sin(theta_min, theta_max)))
        theta_next = jnp.array([theta_min, theta_max]) + self.tau * jnp.concat(setmath.mult([self.alpha_min, self.alpha_max], [u1_min, u1_max]))

        state_next = jnp.vstack((x_next,  # jnp.clip(x_next, self.partition['boundary_jnp'][0][0] + 1e-3, self.partition['boundary_jnp'][1][0] - 1e-3),
                                 y_next,  # jnp.clip(y_next, self.partition['boundary_jnp'][0][1] + 1e-3, self.partition['boundary_jnp'][1][1] - 1e-3),
                                 theta_next))

        state_next_min = jnp.min(state_next, axis=1)
        state_next_max = jnp.max(state_next, axis=1)

        return state_next_min, state_next_max

class DubinsDynamics4D:
    def __init__(self, args):
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None

        # Discretization step size
        self.tau = 0.5

        self.n = 4
        self.p = 2
        self.state_variables = ['x', 'y', 'angle', 'velocity']
        self.wrap = jnp.array([False, False, True, False], dtype=bool)

        if args.model_version == 0:
            logger.info('- Load Dubins without parameter uncertainty')
            # No parameter uncertainty
            self.alpha_min = 0.85
            self.alpha_max = 0.85
            self.alpha = 0.85

            self.beta_min = 0.85
            self.beta_max = 0.85
            self.beta = 0.85
        elif args.model_version == 1:
            logger.info('- Load Dubins with uncertain parameters in the interval [0.80,0.90]')
            # High parameter uncertainty
            self.alpha_min = 0.80
            self.alpha_max = 0.90
            self.alpha = 0.85

            self.beta_min = 0.80
            self.beta_max = 0.90
            self.beta = 0.85
        else:
            logger.info('- Load Dubins with uncertain parameters in the interval [0.75,0.95]')
            # High parameter uncertainty
            self.alpha_min = 0.75
            self.alpha_max = 0.95
            self.alpha = 0.85

            self.beta_min = 0.75
            self.beta_max = 0.95
            self.beta = 0.85

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0, 0, 0.1, 0])**2) # From stdev to covariance
            self.noise.set_partition_probs(num_cells=[1, 1, 10, 1])
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0, 0, 0.1, 0])) # Halfwidth
            self.noise.set_partition_probs(num_cells=[1, 1, 10, 1])
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

    def step(self, state, action, noise):
        x, y, theta, V = state[0], state[1], state[2], state[3]
        u1, u2 = action[0], action[1]
        x_next = x + self.tau * V * jnp.cos(theta)
        y_next = y + self.tau * V * jnp.sin(theta)
        theta_next = wrap_theta(theta + self.tau * self.alpha * u1 + noise[2])
        V_next = self.beta * V + self.tau * u2

        state_next = jnp.array([x_next,
                                y_next,
                                theta_next,
                                jnp.clip(V_next, self.partition['boundary_jnp'][0][3] + 1e-3, self.partition['boundary_jnp'][1][3] - 1e-3)])
        return state_next

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        # Convert to boxes
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [x_min, y_min, theta_min, V_min] = state_min
        [x_max, y_max, theta_max, V_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        [u1_min, u2_min] = jnp.maximum(action_min, self.uMin)
        [u1_max, u2_max] = jnp.minimum(action_max, self.uMax)

        x_next = jnp.array([x_min, x_max]) + self.tau * jnp.concat(setmath.mult([V_min, V_max], setmath.cos(theta_min, theta_max)))
        y_next = jnp.array([y_min, y_max]) + self.tau * jnp.concat(setmath.mult([V_min, V_max], setmath.sin(theta_min, theta_max)))
        theta_next = jnp.array([theta_min, theta_max]) + self.tau * jnp.concat(setmath.mult([self.alpha_min, self.alpha_max], [u1_min, u1_max]))
        V_next = jnp.concat(setmath.mult([self.beta_min, self.beta_max], [V_min, V_max])) + self.tau * jnp.array([u2_min, u2_max])

        state_next = jnp.vstack((x_next,
                                 y_next,
                                 theta_next,
                                 jnp.clip(V_next, self.partition['boundary_jnp'][0][3] + jnp.array([1e-3, 2e-3]), self.partition['boundary_jnp'][1][3] - jnp.array([2e-3, 1e-3]))))

        state_next_min = jnp.min(state_next, axis=1)
        state_next_max = jnp.max(state_next, axis=1)

        return state_next_min, state_next_max

class DroneDynamics:
    def __init__(self, args, dim=2):

        if dim not in [2,3]:
            raise ValueError(f"DroneDynamics only supports dim in [2, 3], got {dim}")

        self.linear = False
        self.independent_state_dims = [[0,1],[2,3]] if dim == 2 else [[0,1],[2,3],[4,5]]
        self.independent_input_dims = [[0],[1]] if dim == 2 else [[0],[1],[2]]

        if dim == 2:
            self.n = 4
            self.p = 2
            self.state_variables = ['x_pos', 'x_vel', 'y_pos', 'y_vel']
            self.wrap = jnp.array([False, False, False, False], dtype=bool)
        else:
            self.n = 6
            self.p = 3
            self.state_variables = ['x_pos', 'x_vel', 'y_pos', 'y_vel', 'z_pos', 'z_vel']
            self.wrap = jnp.array([False, False, False, False, False, False], dtype=bool)

        self.v_min = -2.5
        self.v_max = 2.5

        # Discretization step size
        self.tau = 1.0

        # State transition matrix
        Ablock = np.array([[1, self.tau],
                          [0, 1]])
        
        # Input matrix
        Bblock = np.array([[self.tau**2/2],
                           [self.tau]])
        
        if dim == 2:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock)
            self.B  = scipy.linalg.block_diag(Bblock, Bblock)

            # Disturbance matrix
            self.Q  = np.array([[0],[0],[0],[0]])

            # Covariance of the process noise
            if args.noise_distr == 'gaussian':
                cov = np.array([0.15, 0, 0.15, 0])**2 # From stdev to covariance
                self.noise = GaussianDistr(cov)
                self.noise.set_partition_probs(num_cells=[10, 1, 10, 1])
            elif args.noise_distr == 'triangular':
                cov = np.array([0.15, 0, 0.15, 0]) # Halfwidth
                self.noise = TriangularDistr(cov)
                self.noise.set_partition_probs(num_cells=[10, 1, 10, 1])
            else:
                raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        else:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock, Ablock)
            self.B  = scipy.linalg.block_diag(Bblock, Bblock, Bblock)

            # Disturbance matrix
            self.Q  = np.array([[0],[0],[0],[0], [0], [0]])

            # Covariance of the process noise
            if args.noise_distr == 'gaussian':
                cov = np.array([0.1, 0, 0.1, 0, 0.1, 0])**2 # From stdev to covariance
                self.noise = GaussianDistr(cov)
                self.noise.set_partition_probs(num_cells=[5, 1, 5, 1, 5, 1])
            elif args.noise_distr == 'triangular':
                cov = np.array([0.1, 0, 0.1, 0, 0.1, 0]) # Halfwidth
                self.noise = TriangularDistr(cov)
                self.noise.set_partition_probs(num_cells=[10, 1, 10, 1, 10, 1])
            else:
                raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

    def step(self, state, action, noise):
        state_next = jnp.dot(self.A, state) + jnp.dot(self.B, action) + noise
        state_next = state_next.at[1::2].set(jnp.clip(state_next[1::2], self.v_min + 1e-4, self.v_max - 1e-4))

        return state_next

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):

        action_min = jnp.maximum(action_min, self.uMin)
        action_max = jnp.minimum(action_max, self.uMax)

        # Get vertices of the state and action boxes
        state_vertices = setmath.box2vertices(state_min, state_max)
        action_vertices = setmath.box2vertices(action_min, action_max)
        
        # Propogate dynamics for all vertices
        Ax = jnp.dot(self.A, state_vertices.T).T  # Shape (2^n, n)
        Bu = jnp.dot(self.B, action_vertices.T).T  # Shape (2^p, n)

        # Combine min/max to get the reachable set
        state_next_min = jnp.min(Ax, axis=0) + jnp.min(Bu, axis=0)
        state_next_max = jnp.max(Ax, axis=0) + jnp.max(Bu, axis=0)

        state_next_min = state_next_min.at[1::2].set(jnp.clip(state_next_min[1::2], self.v_min + 1e-4, self.v_max - 1e-4))
        state_next_max = state_next_max.at[1::2].set(jnp.clip(state_next_max[1::2], self.v_min + 1e-4, self.v_max - 1e-4))

        return state_next_min, state_next_max

class DroneDynamics_battery:
    def __init__(self, args, dim=2):

        if dim not in [2,3]:
            raise ValueError(f"DroneDynamics only supports dim in [2, 3], got {dim}")

        self.linear = False
        self.independent_state_dims = None # No independent dimensions due to battery dynamics
        self.independent_input_dims = None

        if dim == 2:
            self.n = 5
            self.p = 2
            self.state_variables = ['x_pos', 'x_vel', 'y_pos', 'y_vel', 'battery']
            self.wrap = jnp.array([False, False, False, False, False], dtype=bool)
            self.pos_idx = [0, 2]
        else:
            self.n = 7
            self.p = 3
            self.state_variables = ['x_pos', 'x_vel', 'y_pos', 'y_vel', 'z_pos', 'z_vel', 'battery']
            self.wrap = jnp.array([False, False, False, False, False, False, False], dtype=bool)
            self.pos_idx = [0, 2, 4]

        self.v_min = -2.5
        self.v_max = 2.5

        # Discretization step size
        self.tau = 1.0

        # State transition matrix
        Ablock = np.array([[1, self.tau],
                          [0, 1]])
        
        # Input matrix
        Bblock = np.array([[self.tau**2/2],
                           [self.tau]])
        
        if dim == 2:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock, 1)
            self.B  = np.zeros((5, 2))
            self.B[:4, :2] = scipy.linalg.block_diag(Bblock, Bblock)

            # Disturbance matrix
            self.Q  = np.array([[0],[0],[0],[0],[0]])

            # Covariance of the process noise
            if args.noise_distr == 'gaussian':
                cov = np.array([0.1, 0, 0.1, 0, 0])**2 # From stdev to covariance
                self.noise = GaussianDistr(cov)
                self.noise.set_partition_probs(num_cells=[10, 1, 10, 1, 1])
            elif args.noise_distr == 'triangular':
                cov = np.array([0.1, 0, 0.1, 0, 0]) # Halfwidth
                self.noise = TriangularDistr(cov)
                self.noise.set_partition_probs(num_cells=[10, 1, 10, 1, 1])
            else:
                raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

        else:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock, Ablock, 1)
            self.B  = np.zeros((7, 3))
            self.B[:6, :3] = scipy.linalg.block_diag(Bblock, Bblock, Bblock)

            # Disturbance matrix
            self.Q  = np.array([[0],[0],[0],[0],[0],[0],[0]])

            # Covariance of the process noise
            if args.noise_distr == 'gaussian':
                cov = np.array([0.1, 0, 0.1, 0, 0.1, 0, 0])**2 # From stdev to covariance
                self.noise = GaussianDistr(cov)
                self.noise.set_partition_probs(num_cells=[5, 1, 5, 1, 5, 1, 1])
            elif args.noise_distr == 'triangular':
                cov = np.array([0.1, 0, 0.1, 0, 0.1, 0, 0]) # Halfwidth
                self.noise = TriangularDistr(cov)
                self.noise.set_partition_probs(num_cells=[10, 1, 10, 1, 10, 1, 1])
            else:
                raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

    def inbox(self, state, box):
        return jnp.all((state >= box[0]) & (state <= box[1]))

    def step(self, state, action, noise):
        state_next = jnp.dot(self.A, state) + jnp.dot(self.B, action) + noise
        state_next = state_next.at[1::2].set(jnp.clip(state_next[1::2], self.v_min + 1e-4, self.v_max - 1e-4))
        battery_idx = self.n - 1

        in_cs = self.inbox(state, self.charging_station[0])
        new_battery = jnp.where(
            in_cs,
            jnp.minimum(state_next[battery_idx] + 10, self.max_charge),
            state_next[battery_idx] - 5
        )
        return state_next.at[battery_idx].set(new_battery)

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):

        action_min = jnp.maximum(action_min, self.uMin)
        action_max = jnp.minimum(action_max, self.uMax)

        # Get vertices of the state and action boxes
        state_vertices = setmath.box2vertices(state_min, state_max)
        action_vertices = setmath.box2vertices(action_min, action_max)
        
        # Propogate dynamics for all vertices
        Ax = jnp.dot(self.A, state_vertices.T).T  # Shape (2^n, n)
        Bu = jnp.dot(self.B, action_vertices.T).T  # Shape (2^p, n)

        # Combine min/max to get the reachable set
        state_next_min = jnp.min(Ax, axis=0) + jnp.min(Bu, axis=0)
        state_next_max = jnp.max(Ax, axis=0) + jnp.max(Bu, axis=0)

        state_next_min = state_next_min.at[1::2].set(jnp.clip(state_next_min[1::2], self.v_min + 1e-4, self.v_max - 1e-4))
        state_next_max = state_next_max.at[1::2].set(jnp.clip(state_next_max[1::2], self.v_min + 1e-4, self.v_max - 1e-4))

        # Battery charging calculation (+10 when charging, capped at max_charge; -5 otherwise)
        cs_min = self.charging_station[0][0]
        cs_max = self.charging_station[0][1]

        entirely_inside = jnp.all((state_min >= cs_min) & (state_max <= cs_max))
        intersects = jnp.all((state_max >= cs_min) & (state_min <= cs_max))

        # entirely_inside: every state charges -> both bounds +10
        # partial overlap: some states charge, some drain -> min -5, max +10
        # no overlap: every state drains -> both bounds -5
        delta_min = jnp.where(entirely_inside, 10.0, -5.0)
        delta_max = jnp.where(intersects, 10.0, -5.0)

        battery_idx = self.n - 1
        new_min = jnp.minimum(state_next_min[battery_idx] + delta_min, self.max_charge)
        new_max = jnp.minimum(state_next_max[battery_idx] + delta_max, self.max_charge)
        state_next_min = state_next_min.at[battery_idx].set(new_min)
        state_next_max = state_next_max.at[battery_idx].set(new_max)

        return state_next_min, state_next_max

class PendulumDynamics:
    def __init__(self, args):
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None

        self.n = 2
        self.p = 1
        self.state_variables = ['angle', 'velocity']
        self.wrap = jnp.array([True, False], dtype=bool)

        # Discretization step size
        self.tau = 0.05 * 2

        # Pendulum parameters
        self.G = 10
        self.m = 1.0
        self.l = 1.0
        self.b = 0.0 # Gymnasium pendulum does not have damping

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.03, 0.1])**2) # From stdev to covariance
            self.noise.set_partition_probs(num_cells=[10, 10])
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.03, 0.1])) # Halfwidth
            self.noise.set_partition_probs(num_cells=[10, 10])
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

    def step(self, state, action, noise):

        new_velo = (1 - self.b) * state[1] + \
                   (3 * self.G / (2 * self.l) * jnp.sin(state[0])) * self.tau + \
                   (3.0 / (self.m * self.l**2) * action[0]) * self.tau
        new_angle = wrap_theta(state[0] + self.tau * new_velo + noise[0])
        new_velo = new_velo + noise[1]

        return jnp.array([new_angle, new_velo])

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [angle_min, velo_min] = state_min
        [angle_max, velo_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        u_min = jnp.maximum(action_min, self.uMin)[0]
        u_max = jnp.minimum(action_max, self.uMax)[0]

        velo_next = setmath.tuple2box((1 - self.b) * jnp.array([velo_min, velo_max])) + \
                    setmath.tuple2box(self.tau * 3 * self.G / (2 * self.l) * setmath.tuple2box(setmath.sin(angle_min, angle_max))) + \
                    setmath.tuple2box(self.tau * 3.0 / (self.m * self.l ** 2) * jnp.array([u_min, u_max]))
        
        angle_next = jnp.array([angle_min, angle_max]) + self.tau * velo_next

        state_next = jnp.vstack((angle_next,
                                 velo_next))

        state_next_min = jnp.min(state_next, axis=1)
        state_next_max = jnp.max(state_next, axis=1)

        return state_next_min, state_next_max
    
class MountainCarDynamics:
    def __init__(self, args):
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None

        self.n = 2
        self.p = 1
        self.state_variables = ['position', 'velocity']
        self.wrap = jnp.array([False, False], dtype=bool)

        # Discretization step size
        self.tau = 1

        # Parameters
        self.max_speed = 0.07
        self.gravity = 0.0025
        self.power = 0.0015

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.005, 0.0005])**2) # From stdev to covariance
            self.noise.set_partition_probs(num_cells=[10, 10])
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.005, 0.0005])) # Halfwidth
            self.noise.set_partition_probs(num_cells=[10, 10])
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')
    def step(self, state, action, noise):

        position, velocity = state[0], state[1]

        velocity = velocity + self.tau * (action[0] * self.power - self.gravity * jnp.cos(3 * position))
        velocity = jnp.clip(velocity, -self.max_speed+1e-4, self.max_speed-1e-4)
        position = position + self.tau * velocity + noise[0]
        velocity = velocity + noise[1]

        return jnp.array([position, velocity])

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [pos_min, velo_min] = state_min
        [pos_max, velo_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        u_min = jnp.maximum(action_min, self.uMin)[0]
        u_max = jnp.minimum(action_max, self.uMax)[0]

        velo_next = jnp.array([velo_min, velo_max]) + \
                    setmath.tuple2box(self.tau * -self.gravity * setmath.tuple2box(setmath.cos(3 * pos_min, 3 * pos_max)) ) + \
                    setmath.tuple2box(self.tau * self.power * jnp.array([u_min, u_max]))

        # Match the real dynamics (see step): velocity saturates at the speed limit before it
        # drives the position update. The clip is monotone, so applying it to the [lb, ub]
        # interval is exact. Without it, high-velocity cells (the goal-approach corridor) overshoot
        # the velocity grid boundary (±max_speed) and get mapped to the absorbing/failure state.
        velo_next = jnp.clip(velo_next, -self.max_speed + 1e-4, self.max_speed - 1e-4)

        pos_next = jnp.array([pos_min, pos_max]) + self.tau * velo_next

        state_next = jnp.vstack((pos_next,
                                 velo_next))

        state_next_min = jnp.min(state_next, axis=1)
        state_next_max = jnp.max(state_next, axis=1)

        return state_next_min, state_next_max
    
class CartPoleDynamics:
    def __init__(self, args):
        '''
        CartPole dynamics model, based on the Gymnasium CartPole environment. 
        The state consists of the cart position, cart velocity, pole angle, and pole angular velocity. 
        The action is a force applied to the cart. The dynamics are nonlinear and include process noise.
        
        Differences with the original CartPole environment:
        - The time step is 0.02 * 5 seconds instead of 0.02 seconds.
        - Original environment has no process noise.
        - Original environment has a discrete action space (left/right), while this model uses a continuous action space (force).
        '''
        
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None

        self.n = 4
        self.p = 1
        self.state_variables = ['position', 'velocity', 'angle', 'angular_velocity']
        self.wrap = jnp.array([False, False, False, False], dtype=bool)

        # Discretization step size
        self.tau = 0.02 * 5

        # CartPole parameters (Gymnasium CartPole convention)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # Actually half the pole's length
        self.polemass_length = self.masspole * self.length

        # Covariance of the process noise (on cart position and pole angle)
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0, 0.001, 0, 0])**2) # From stdev to covariance
            self.noise.set_partition_probs(num_cells=[1, 1, 1, 1])
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.005, 0, 0.005, 0])) # Halfwidth
            self.noise.set_partition_probs(num_cells=[10, 1, 10, 1])
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

    def step(self, state, action, noise):
        x, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]
        force = action[0]

        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        return jnp.array([x, x_dot, theta, theta_dot]) + noise

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [x_min, xd_min, th_min, thd_min] = state_min
        [x_max, xd_max, th_max, thd_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        F_min = jnp.maximum(action_min, self.uMin)[0]
        F_max = jnp.minimum(action_max, self.uMax)[0]

        # Interval bounds on the trigonometric terms of the pole angle
        s_min, s_max = setmath.sin(th_min, th_max)
        c_min, c_max = setmath.cos(th_min, th_max)

        # theta_dot^2
        thd2_min, thd2_max = setmath.square(thd_min, thd_max)

        # temp = (F + polemass_length * theta_dot^2 * sin(theta)) / total_mass
        # (setmath.mult/div return shape-(1,) arrays, so we squeeze back to scalars)
        q_min, q_max = setmath.mult((thd2_min, thd2_max), (s_min, s_max))
        q_min, q_max = q_min[0], q_max[0]
        temp_min = (F_min + self.polemass_length * q_min) / self.total_mass
        temp_max = (F_max + self.polemass_length * q_max) / self.total_mass

        # thetaacc = (g*sin - cos*temp) / (length * (4/3 - masspole*cos^2/total_mass))
        b_min, b_max = setmath.mult((c_min, c_max), (temp_min, temp_max))
        b_min, b_max = b_min[0], b_max[0]
        num_min = self.gravity * s_min - b_max
        num_max = self.gravity * s_max - b_min

        c2_min, c2_max = setmath.square(c_min, c_max)
        k = self.masspole / self.total_mass
        # Denominator is guaranteed positive (4/3 - k*cos^2 >= 4/3 - k > 0)
        denom_min = self.length * (4.0 / 3.0 - k * c2_max)
        denom_max = self.length * (4.0 / 3.0 - k * c2_min)

        thacc_min, thacc_max = setmath.div((num_min, num_max), (denom_min, denom_max))
        thacc_min, thacc_max = thacc_min[0], thacc_max[0]

        # xacc = temp - polemass_length * thetaacc * cos / total_mass
        e_min, e_max = setmath.mult((thacc_min, thacc_max), (c_min, c_max))
        e_min, e_max = e_min[0], e_max[0]
        coef = self.polemass_length / self.total_mass
        xacc_min = temp_min - coef * e_max
        xacc_max = temp_max - coef * e_min

        # Euler integration
        x_next = jnp.array([x_min + self.tau * xd_min, x_max + self.tau * xd_max])
        xd_next = jnp.array([xd_min + self.tau * xacc_min, xd_max + self.tau * xacc_max])
        th_next = jnp.array([th_min + self.tau * thd_min, th_max + self.tau * thd_max])
        thd_next = jnp.array([thd_min + self.tau * thacc_min, thd_max + self.tau * thacc_max])

        state_next = jnp.vstack((x_next, xd_next, th_next, thd_next))

        state_next_min = jnp.min(state_next, axis=1)
        state_next_max = jnp.max(state_next, axis=1)

        return state_next_min, state_next_max

class DoubleIntegratorDynamics:
    def __init__(self, args):
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None

        self.n = 2
        self.p = 1
        self.state_variables = ['position', 'velocity']
        self.wrap = jnp.array([True, False], dtype=bool)

        # Discretization step size
        self.tau = 1.0

        # State transition matrix
        self.A  = np.array([[1, self.tau],
                          [0, 1]])
        
        # Input matrix
        self.B  = np.array([[self.tau**2/2],
                           [self.tau]])
    
        # Disturbance matrix
        self.Q  = np.array([[0],[0],])

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.15, 0.15])**2) # From stdev to covariance
            self.noise.set_partition_probs(num_cells=[10, 10])
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.15, 0.15])) # Halfwidth
            self.noise.set_partition_probs(num_cells=[10, 10])
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

    def step(self, state, action, noise):
        state_next = jnp.dot(self.A, state) + jnp.dot(self.B, action) + noise

        return state_next

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):

        action_min = jnp.maximum(action_min, self.uMin)
        action_max = jnp.minimum(action_max, self.uMax)

        # Get vertices of the state and action boxes
        state_vertices = setmath.box2vertices(state_min, state_max)
        action_vertices = setmath.box2vertices(action_min, action_max)
        
        # Propogate dynamics for all vertices
        Ax = jnp.dot(self.A, state_vertices.T).T  # Shape (2^n, n)
        Bu = jnp.dot(self.B, action_vertices.T).T  # Shape (2^p, n)

        # Combine min/max to get the reachable set
        state_next_min = jnp.min(Ax, axis=0) + jnp.min(Bu, axis=0)
        state_next_max = jnp.max(Ax, axis=0) + jnp.max(Bu, axis=0)

        return state_next_min, state_next_max
    
class Test1DDynamics:
    def __init__(self, args):
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None

        self.n = 1
        self.p = 1
        self.state_variables = ['position']
        self.wrap = jnp.array([True], dtype=bool)

        # State transition matrix
        self.A  = np.array([[1]])
        
        # Input matrix
        self.B  = np.array([[1]])

        # Covariance of the process noise
        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.2])**2) # From stdev to covariance
            self.noise.set_partition_probs(num_cells=[10])
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.2])) # Halfwidth
            self.noise.set_partition_probs(num_cells=[10])
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

    def step(self, state, action, noise):
        state_next = jnp.dot(self.A, state) + jnp.dot(self.B, action) + noise

        return state_next

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):

        action_min = jnp.maximum(action_min, self.uMin)
        action_max = jnp.minimum(action_max, self.uMax)

        # Get vertices of the state and action boxes
        state_vertices = setmath.box2vertices(state_min, state_max)
        action_vertices = setmath.box2vertices(action_min, action_max)
        
        # Propogate dynamics for all vertices
        Ax = jnp.dot(self.A, state_vertices.T).T  # Shape (2^n, n)
        Bu = jnp.dot(self.B, action_vertices.T).T  # Shape (2^p, n)

        # Combine min/max to get the reachable set
        state_next_min = jnp.min(Ax, axis=0) + jnp.min(Bu, axis=0)
        state_next_max = jnp.max(Ax, axis=0) + jnp.max(Bu, axis=0)

        return state_next_min, state_next_max