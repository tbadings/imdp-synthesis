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
    return (theta + np.pi) % (2 * np.pi) - np.pi

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
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0, 0, 0.2])) # Halfwidth
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

    def step(self, state, action, noise):
        [x, y, theta] = state
        [u1, u2] = action
        x_next = x + self.tau * u2 * np.cos(theta)
        y_next = y + self.tau * u2 * np.sin(theta)
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
        self.wrap = jnp.array([False, False, False, False], dtype=bool)

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
        self.noise = GaussianDistr(np.array([0, 0, 0.1, 0])**2) # From stdev to covariance

    def step(self, state, action, noise):
        [x, y, theta, V] = state
        [u1, u2] = action
        x_next = x + self.tau * V * np.cos(theta)
        y_next = y + self.tau * V * np.sin(theta)
        theta_next = wrap_theta(theta + self.tau * self.alpha * u1 + noise[2])
        V_next = self.beta * V + self.tau * u2

        state_next = jnp.array([x_next,
                                y_next,
                                theta_next,
                                np.clip(V_next, self.partition['boundary_jnp'][0][3] + 1e-3, self.partition['boundary_jnp'][1][3] - 1e-3)])
        return state_next

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        # Convert to boxes
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [x_min, y_min, theta_min, V_min] = state_min
        [x_max, y_max, theta_max, V_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        [u1_min, u2_min] = jnp.maximum(action_min, self.uMin)
        [u1_max, u2_max] = jnp.minimum(action_min, self.uMax)

        Vmean = (V_max + V_min) / 2
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
            assert False

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
            elif args.noise_distr == 'triangular':
                cov = np.array([0.15, 0, 0.15, 0]) # Halfwidth
                self.noise = TriangularDistr(cov)

        else:
            self.A  = scipy.linalg.block_diag(Ablock, Ablock, Ablock)
            self.B  = scipy.linalg.block_diag(Bblock, Bblock, Bblock)

            # Disturbance matrix
            self.Q  = np.array([[0],[0],[0],[0], [0], [0]])

            # Covariance of the process noise
            if args.noise_distr == 'gaussian':
                cov = np.array([0.15, 0, 0.15, 0, 0.15, 0])**2 # From stdev to covariance
                self.noise = GaussianDistr(cov)
            elif args.noise_distr == 'triangular':
                cov = np.array([0.15, 0, 0.15, 0, 0.15, 0]) # Halfwidth
                self.noise = TriangularDistr(cov)

        self.noise = GaussianDistr(cov)

    def step(self, state, action, noise):
        state_next = self.A @ state + self.B @ action + noise

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
        self.noise = GaussianDistr(np.array([0.03, 0.1])**2) # From stdev to covariance

    def step(self, state, action, noise):

        new_velo = (1 - self.b) * state[1] + \
                   (3 * self.G / (2 * self.l) * np.sin(state[0])) * self.tau + \
                   (3.0 / (self.m * self.l**2) * action[0]) * self.tau
        new_angle = wrap_theta(state[0] + self.tau * new_velo + noise[0])
        new_velo = new_velo + noise[1]

        return np.array([new_angle, new_velo])

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
        self.tau = 2

        # Parameters
        self.max_speed = 0.07
        self.gravity = 0.0025
        self.power = 0.0015

        # Covariance of the process noise
        self.noise = GaussianDistr(np.array([0.005,0.0005])**2) # From stdev to covariance

    def step(self, state, action, noise):

        position, velocity = state

        velocity = velocity + self.tau * (action[0] * self.power - self.gravity * np.cos(3 * position))
        velocity = np.clip(velocity, -self.max_speed+1e-4, self.max_speed-1e-4)
        position += self.tau * velocity + noise[0]
        velocity += noise[1]

        return np.array([position, velocity])

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
        
        pos_next = jnp.array([pos_min, pos_max]) + self.tau * velo_next

        state_next = jnp.vstack((pos_next,
                                 velo_next))

        state_next_min = jnp.min(state_next, axis=1)
        state_next_max = jnp.max(state_next, axis=1)

        return state_next_min, state_next_max

class CartpoleDynamics:
    def __init__(self, args):
        self.linear = False
        self.independent_state_dims = None
        self.independent_input_dims = None

        self.n = 4
        self.p = 1
        self.state_variables = ['position', 'velocity', 'angle', 'angular_velocity']
        self.wrap = jnp.array([False, False, True, False], dtype=bool)

        # Gymnasium CartPole-v1 physical constants.
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.angle_offset = np.pi

        if args.noise_distr == 'gaussian':
            self.noise = GaussianDistr(np.array([0.005, 0.02, 0.005, 0.02])**2)
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([0.01, 0.04, 0.01, 0.04]))
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

    def _accelerations(self, state, action):
        _, x_dot, theta, theta_dot = state
        theta = theta + self.angle_offset
        force = np.clip(action[0], -self.force_mag, self.force_mag)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        return xacc, thetaacc

    def step(self, state, action, noise):
        x, x_dot, theta, theta_dot = state
        xacc, thetaacc = self._accelerations(state, action)

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        return np.array([x, x_dot, theta, theta_dot]) + noise

    @partial(jax.jit, static_argnums=(0))
    def step_set(self, state_min, state_max, action_min, action_max):
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        [x_min, xdot_min, theta_min, thetadot_min] = state_min
        [x_max, xdot_max, theta_max, thetadot_max] = state_max

        force_min = jnp.maximum(action_min, self.uMin)[0]
        force_max = jnp.minimum(action_max, self.uMax)[0]

        sin_theta = setmath.sin(theta_min + self.angle_offset, theta_max + self.angle_offset)
        cos_theta = setmath.cos(theta_min + self.angle_offset, theta_max + self.angle_offset)
        cos_sq = self._square_interval(cos_theta[0], cos_theta[1])
        theta_dot_sq = self._square_interval(thetadot_min, thetadot_max)

        temp = self._div_interval(
            *self._add_interval(
                (force_min, force_max),
                self._mul_interval(
                    (self.polemass_length * theta_dot_sq[0], self.polemass_length * theta_dot_sq[1]),
                    sin_theta,
                ),
            ),
            self.total_mass,
            self.total_mass,
        )

        denom = self._mul_interval(
            (self.length, self.length),
            self._add_interval(
                (4.0 / 3.0, 4.0 / 3.0),
                (-self.masspole * cos_sq[1] / self.total_mass, -self.masspole * cos_sq[0] / self.total_mass),
            ),
        )
        thetaacc = self._div_interval(
            *self._add_interval(
                (self.gravity * sin_theta[0], self.gravity * sin_theta[1]),
                self._mul_interval((-cos_theta[1], -cos_theta[0]), temp),
            ),
            denom[0],
            denom[1],
        )
        xacc = self._add_interval(
            temp,
            self._mul_interval(
                (-self.polemass_length / self.total_mass * thetaacc[1],
                 -self.polemass_length / self.total_mass * thetaacc[0]),
                cos_theta,
            ),
        )

        xdot_next = self._add_interval((xdot_min, xdot_max), (self.tau * xacc[0], self.tau * xacc[1]))
        x_next = self._add_interval((x_min, x_max), (self.tau * xdot_min, self.tau * xdot_max))
        thetadot_next = self._add_interval((thetadot_min, thetadot_max), (self.tau * thetaacc[0], self.tau * thetaacc[1]))
        theta_next = self._add_interval((theta_min, theta_max), (self.tau * thetadot_min, self.tau * thetadot_max))

        state_next_min = jnp.array([
            jnp.squeeze(x_next[0]),
            jnp.squeeze(xdot_next[0]),
            jnp.squeeze(theta_next[0]),
            jnp.squeeze(thetadot_next[0]),
        ])
        state_next_max = jnp.array([
            jnp.squeeze(x_next[1]),
            jnp.squeeze(xdot_next[1]),
            jnp.squeeze(theta_next[1]),
            jnp.squeeze(thetadot_next[1]),
        ])
        return state_next_min, state_next_max

    @staticmethod
    def _add_interval(x, y):
        return x[0] + y[0], x[1] + y[1]

    @staticmethod
    def _mul_interval(x, y):
        z_min, z_max = setmath.mult(jnp.array([x[0], x[1]]), jnp.array([y[0], y[1]]))
        return jnp.squeeze(z_min), jnp.squeeze(z_max)

    @staticmethod
    def _div_interval(x_min, x_max, y_min, y_max):
        return setmath.mult(jnp.array([x_min, x_max]), jnp.array([1.0 / y_max, 1.0 / y_min]))

    @staticmethod
    def _square_interval(x_min, x_max):
        sq_min = jnp.where((x_min <= 0) & (x_max >= 0), 0.0, jnp.minimum(x_min**2, x_max**2))
        sq_max = jnp.maximum(x_min**2, x_max**2)
        return sq_min, sq_max
    
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
        self.noise = GaussianDistr(np.array([0.15, 0.15])**2) # From stdev to covariance

    def step(self, state, action, noise):
        state_next = self.A @ state + self.B @ action + noise

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
        elif args.noise_distr == 'triangular':
            self.noise = TriangularDistr(np.array([1])) # Halfwidth
        else:
            raise ValueError(f'Unsupported noise distribution: {args.noise_distr}. Expected "gaussian" or "triangular".')

    def step(self, state, action, noise):
        state_next = self.A @ state + self.B @ action + noise

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
