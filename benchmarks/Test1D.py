from functools import partial
from benchmarks.models import Test1DDynamics
import jax
import jax.numpy as jnp
import numpy as np
from benchmarks.dynamics import setmath


class Test1D(Test1DDynamics):
    '''
    Dubin's vehicle benchmark, with a 4D state space and a 2D control input space.
    '''

    def __init__(self, args):
        Test1DDynamics.__init__(self, args)

        self.plot_dimensions = [0]

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
        self.uMin = [-1]
        self.uMax = [1]
        self.num_actions = [3]

        self.partition['boundary'] = np.array([[-5], [5]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([10])

        self.goal = np.array([
            [[-1], [1]]
        ], dtype=float)

        self.critical = np.array([
        ], dtype=float)

        self.x0 = np.array([-2.5])

        return
