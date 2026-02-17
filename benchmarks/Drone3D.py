from functools import partial
from benchmarks.models import DroneDynamics
import jax
import jax.numpy as jnp
import numpy as np
import scipy 
from core import setmath


class Drone3D(DroneDynamics):
    '''
    Drone benchmark, with a 6D state space and a 3D control input space.
    '''

    def __init__(self, args):
        DroneDynamics.__init__(self, args, dim=3)

        self.plot_dimensions = [0, 2]

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
        self.uMin = [-3, -3, -3]
        self.uMax = [3, 3, 3]
        self.num_actions = [5, 5, 5]

        v_min = -3.5
        v_max = 3.5

        self.partition['boundary'] = np.array([[-15, v_min, -9, v_min, -7, v_min], 
                                               [15, v_max, 9, v_max, 7, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([30, 7, 18, 7, 14, 7])

        self.goal = np.array([
            # [[11, v_min, 1, v_min, -7, v_min], [15, v_max, 5, v_max, -3, v_max]]
            [[11, v_min, 1, v_min, 0, v_min], [15, v_max, 5, v_max, 7, v_max]]
        ], dtype=float)

        self.critical = np.array([
            # Hole 1
            [[-11, v_min, -1, v_min, -7, v_min], [-5, v_max, 9, v_max, -5, v_max]],
            [[-11, v_min, 5, v_min, -5, v_min], [-5, v_max, 9, v_max, 5, v_max]],
            [[-11, v_min, -1, v_min, -5, v_min], [-5, v_max, 3, v_max, 3, v_max]],

            # Hole 2
            [[-1, v_min, 1, v_min, -7, v_min], [3, v_max, 9, v_max, -1, v_max]],
            [[-1, v_min, 1, v_min, 3, v_min], [3, v_max, 9, v_max, 5, v_max]],
            [[-1, v_min, 1, v_min, -1, v_min], [3, v_max, 3, v_max, 3, v_max]],
            [[-1, v_min, 7, v_min, -1, v_min], [3, v_max, 9, v_max, 3, v_max]],

            # Tower
            [[-1, v_min, -3, v_min, -7, v_min], [3, v_max, 1, v_max, 7, v_max]],

            # Wall between routes
            [[3, v_min, -3, v_min, -7, v_min], [9, v_max, 1, v_max, -1, v_max]],

            # Long route obstacles
            [[-11, v_min, -5, v_min, -7, v_min], [-7, v_max, -1, v_max, 1, v_max]],
            [[-1, v_min, -9, v_min, -7, v_min], [3, v_max, -3, v_max, -5, v_max]],

            # Overhanging
            [[-1, v_min, -9, v_min, 3, v_min], [3, v_max, -3, v_max, 7, v_max]],

            # Small last obstacle
            [[11, v_min, -9, v_min, -7, v_min], [15, v_max, -5, v_max, -5, v_max]],

            # Obstacle next to goal
            [[9, v_min, 5, v_min, -7, v_min], [15, v_max, 9, v_max, 1, v_max]],
        ], dtype=float)

        self.x0 = np.array([-14.5, 0, 6, 0, 2, 0])

        return



class Drone3D_small(DroneDynamics):
    '''
    Drone benchmark, with a 6D state space and a 3D control input space.
    '''

    def __init__(self, args):
        DroneDynamics.__init__(self, args, dim=3)

        self.plot_dimensions = [0, 2]

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
        self.uMin = [-3, -3, -3]
        self.uMax = [3, 3, 3]
        self.num_actions = [5, 5, 5]

        v_min = -3.5
        v_max = 3.5

        self.partition['boundary'] = np.array([[-5, v_min, -5, v_min, -5, v_min],
                                               [5, v_max, 5, v_max, 5, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([10, 7, 10, 7, 10, 7])

        self.goal = np.array([
            [[1, v_min, 1, v_min, 1, v_min], [5, v_max, 5, v_max, 5, v_max]],
        ], dtype=float)

        self.critical = np.array([
        ], dtype=float)

        self.x0 = np.array([-3.5, 0, -3.5, 0, -3.5, 0])

        return
