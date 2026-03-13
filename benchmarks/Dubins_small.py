from functools import partial

from benchmarks.models import DubinsSmallDynamics
import jax
import jax.numpy as jnp
import numpy as np

from core import setmath


def wrap_theta(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


class Dubins_small(DubinsSmallDynamics):
    '''
    Simplified version of the Dubin's vehicle benchmark, with a 3D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DubinsSmallDynamics.__init__(self, args)
        
        self.plot_dimensions = [0, 1]

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 1

        self.set_spec()
        print('')

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-0.50 * np.pi, -3]
        self.uMax = [0.50 * np.pi, 3]
        self.num_actions = [7, 5]

        self.partition['boundary'] = np.array([[-15, -10, -np.pi], [10, 10, np.pi]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([40, 40, 31])

        # self.goal = np.array([
        #     [[-10, 5, -np.pi], [-5, 10, np.pi]]
        # ], dtype=float)

        self.goal = np.array([
            [[7, -10, -np.pi], [10, 10, np.pi]]
        ], dtype=float)

        # self.critical = np.array([
        #     [[-10, -1, -np.pi], [-1, 1, np.pi]],
        #     [[-1, -5, -np.pi], [1, 4, np.pi]]
        # ], dtype=float)

        self.critical = np.array([
            # [[-1, 9, -np.pi], [1, 10, np.pi]],
            [[-1, -4, -np.pi], [1, 4.5, np.pi]],
            # [[-6, -1, -np.pi], [-1, 2, np.pi]],
            # [[-1, -10, -np.pi], [1, -9, np.pi]]
        ], dtype=float)

        # self.x0 = np.array([-7.5, -7.5, 0])

        self.x0 = np.array([-10, 0, 0])

        return