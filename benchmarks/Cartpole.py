from benchmarks.models import CartpoleDynamics
import jax.numpy as jnp
import numpy as np


class Cartpole(CartpoleDynamics):
    '''
    Cart-pole swing-up benchmark.
    '''

    def __init__(self, args):
        CartpoleDynamics.__init__(self, args)

        self.plot_dimensions = [0, 2]

        # Nonlinear benchmark; no action lumping is applied by the parser.
        self.lump = 1

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        self.uMin = [-self.force_mag]
        self.uMax = [self.force_mag]
        self.num_actions = [7]

        x_threshold = 2.4

        self.partition['boundary'] = np.array([
            [-x_threshold, -4.0, -np.pi, -8.0],
            [ x_threshold,  4.0,  np.pi,  8.0],
        ])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([21, 21, 49, 21])

        self.goal = np.array([
            [[-0.7, -1.5, -np.pi, -2.5],
             [ 0.7,  1.5, -2.65,  2.5]],
            [[-0.7, -1.5,  2.65, -2.5],
             [ 0.7,  1.5,  np.pi,  2.5]],
        ], dtype=float)

        self.critical = np.array([
        ], dtype=float)

        self.x0 = np.array([0.0, 0.0, 0.0, 0.0])

        return
