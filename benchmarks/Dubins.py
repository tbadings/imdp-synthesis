from functools import partial
from benchmarks.models import DubinsDynamics
import jax
import jax.numpy as jnp
import numpy as np
from benchmarks.dynamics import setmath


class Dubins(DubinsDynamics):
    '''
    Dubin's vehicle benchmark, with a 4D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DubinsDynamics.__init__(self, args)

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
        self.uMin = [-0.5 * np.pi, -5]
        self.uMax = [0.5 * np.pi, 5]
        self.num_actions = [7, 7]

        self.partition['boundary'] = np.array([[-10, 0, -np.pi, -3], [10, 10, np.pi, 3]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([40, 20, 20, 20])

        self.goal = np.array([
            [[6, 6, -np.pi, -3], [9, 9, np.pi, 3]]
        ], dtype=float)

        self.critical = np.array([
            [[4, 5, -2 * np.pi, -3], [5, 10, 2 * np.pi, 3]],
            [[-1, 0, -2 * np.pi, -3], [0, 5, 2 * np.pi, 3]],
            [[-5, 4, -2 * np.pi, -3], [-1, 5, 2 * np.pi, 3]],
        ], dtype=float)

        self.x0 = np.array([-3, 2, 0, 0])

        return
