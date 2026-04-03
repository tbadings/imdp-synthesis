from functools import partial
from benchmarks.models import DroneDynamics
import jax
import jax.numpy as jnp
import numpy as np
import scipy 
from benchmarks.dynamics import setmath


class Drone2D(DroneDynamics):
    '''
    Drone benchmark, with a 4D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DroneDynamics.__init__(self, args)

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
        self.uMin = [-3, -3]
        self.uMax = [3 ,3]
        self.num_actions = [5, 5]

        v_min = -3.5 # -3.5 not enough (given 0.50 satprob)
        v_max = 3.5

        self.partition['boundary'] = np.array([[-7, v_min, -7, v_min], [7, v_max, 7, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([14, 7, 14, 7]) # 7 not enough

        self.goal = np.array([
            [[3, v_min, 3, v_min], [6, v_max, 6, v_max]]
        ], dtype=float)

        self.critical = np.array([
            [[-7, v_min, 1, v_min], [-1, v_max, 3, v_max]],
            [[3, v_min, -7, v_min], [7, v_max, -3, v_max]],
        ], dtype=float)

        self.x0 = np.array([-5.5, 0, -5.5, 0])

        return
