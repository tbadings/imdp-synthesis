from functools import partial
from benchmarks.models import DroneDynamics, DroneDynamics_battery
import jax
import jax.numpy as jnp
import numpy as np
import scipy 
from benchmarks.dynamics import setmath


class Drone4D(DroneDynamics):
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

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-1, -1]
        self.uMax = [1 ,1]
        self.num_actions = [5, 5]

        v_min = -3.5 # -3.5 not enough (given 0.50 satprob)
        v_max = 3.5

        self.partition['boundary'] = np.array([[-7, v_min, -7, v_min], [7, v_max, 7, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([28, 14, 28, 14]) # 7 not enough

        self.goal = np.array([
            [[3, v_min, 3, v_min], [7, v_max, 7, v_max]]
        ], dtype=float)

        self.critical = np.array([
            [[-7, v_min, 1, v_min], [-1, v_max, 3, v_max]],
            [[3, v_min, -7, v_min], [7, v_max, -3, v_max]],
        ], dtype=float)

        self.x0 = np.array([-5.5, 0.01, -5.5, 0.01])

        self.pi_arch = [128, 128]
        self.vf_arch = [128, 128]
        self.inflation_rate = [(-2, 2), (-1, 1), (-2, 2), (-1, 1)]

        return

class Drone4D_battery(DroneDynamics_battery):
    '''
    Drone benchmark, with a 4D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DroneDynamics_battery.__init__(self, args, dim=2)

        self.plot_dimensions = [0, 2]

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 1

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-1, -1]
        self.uMax = [1 ,1]
        self.num_actions = [5, 5]

        v_min = -3.5 # -3.5 not enough (given 0.50 satprob)
        v_max = 3.5

        self.partition['boundary'] = np.array([[-10, v_min, -10, v_min, 0], [10, v_max, 10, v_max, 100]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([28, 14, 28, 14, 30]) # 7 not enough
        
        self.goal = np.array([
            [[7, v_min, 7, v_min, 0], [10, v_max, 10, v_max, 100]]
        ], dtype=float)

        self.critical = np.array([
            # [[-7, v_min, 1, v_min, 0], [-1, v_max, 3, v_max, 100]],
            # [[3, v_min, -7, v_min, 0], [7, v_max, -3, v_max, 100]],
        ], dtype=float)

        self.charging_station = np.array([
            [[-8.5, v_min, 5.0, v_min], [-5, v_max, 8.5, v_max]]
        ], dtype=float)

        self.x0 = np.array([-9.5, 0.01, -9.5, 0.01, 50])

        self.pi_arch = [64, 64]
        self.vf_arch = [64, 64]
        self.inflation_rate = [(-2, 2), (-1, 1), (-2, 2), (-1, 1), (-1, 1)]

        return

