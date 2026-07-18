from functools import partial
from benchmarks.models import DroneDynamics, DroneDynamics_battery
import jax
import jax.numpy as jnp
import numpy as np
import scipy 
from benchmarks.dynamics import setmath


class Drone6D(DroneDynamics):
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

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-1, -1, -1]
        self.uMax = [1, 1, 1]
        self.num_actions = [5, 5, 5]

        v_min = -3.5
        v_max = 3.5

        self.partition['boundary'] = np.array([[-17, v_min, -9, v_min, -7, v_min], 
                                               [17, v_max, 9, v_max, 7, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([68, 14, 36, 14, 28, 14])

        self.goal = np.array([
            [[11, v_min, 1, v_min, -7, v_min], [15, v_max, 5, v_max, -3, v_max]]
        ], dtype=float)

        self.critical = np.array([
            # Hole 1
            [[-11, v_min, -1, v_min, -7, v_min], [-5, v_max, 9, v_max, -5, v_max]],
            [[-11, v_min, 5, v_min, -5, v_min], [-5, v_max, 9, v_max, 5, v_max]],
            [[-11, v_min, -1, v_min, -5, v_min], [-5, v_max, 3, v_max, 3, v_max]],

            # # Hole 2
            [[-1, v_min, 1, v_min, -7, v_min], [3, v_max, 9, v_max, -1, v_max]],
            [[-1, v_min, 1, v_min, 3, v_min], [3, v_max, 9, v_max, 5, v_max]],
            [[-1, v_min, 1, v_min, -1, v_min], [3, v_max, 3, v_max, 3, v_max]],
            [[-1, v_min, 7, v_min, -1, v_min], [3, v_max, 9, v_max, 3, v_max]],

            # # Tower
            [[-1, v_min, -3, v_min, -7, v_min], [3, v_max, 1, v_max, 7, v_max]],

            # # Wall between routes
            [[3, v_min, -3, v_min, -7, v_min], [9, v_max, 1, v_max, -1, v_max]],

            # # Long route obstacles
            # [[-11, v_min, -5, v_min, -7, v_min], [-7, v_max, -1, v_max, 1, v_max]],
            [[-1, v_min, -9, v_min, -7, v_min], [3, v_max, -3, v_max, -5, v_max]],

            # Overhanging
            [[-1, v_min, -9, v_min, 3, v_min], [3, v_max, -3, v_max, 7, v_max]],

            # Small last obstacle
            [[11, v_min, -9, v_min, -7, v_min], [15, v_max, -5, v_max, -5, v_max]],

            # Obstacle next to goal
            [[9, v_min, 5, v_min, -7, v_min], [15, v_max, 9, v_max, 1, v_max]],
        ], dtype=float)

        self.x0 = np.array([-14.5, 0.01, 6, 0.01, 2, 0.01])

        self.pi_arch = [64, 64]
        self.vf_arch = [64, 64]
        self.inflation_rate = [(-3, 3), (-1, 1), (-3, 3), (-1, 1), (-3, 3), (-1, 1)]

        return



class Drone6D_small(DroneDynamics):
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

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-1, -1, -1]
        self.uMax = [1, 1, 1]
        self.num_actions = [5, 5, 5]

        v_min = -3.5 # -3.5 not enough (given 0.50 satprob)
        v_max = 3.5

        self.partition['boundary'] = np.array([[-7, v_min, -7, v_min, -7, v_min], [7, v_max, 7, v_max, 7, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([28, 14, 28, 14, 28, 14])

        self.goal = np.array([
            [[3, v_min, 3, v_min, -7, v_min], [7, v_max, 7, v_max, 7, v_max]]
        ], dtype=float)

        self.critical = np.array([
            [[-7, v_min, 1, v_min, -7, v_min], [-1, v_max, 3, v_max, 7, v_max]],
            [[3, v_min, -7, v_min, -7, v_min], [7, v_max, -3, v_max, 7, v_max]],
        ], dtype=float)

        self.x0 = np.array([-5.5, 0.01, -5.5, 0.01, 0.01, 0.01])

        self.pi_arch = [32, 32]
        self.vf_arch = [32, 32]
        self.inflation_rate = [(-2, 2), (-1, 1), (-2, 2), (-1, 1), (-2, 2), (-1, 1)]

        return


class Drone6D_battery(DroneDynamics_battery):
    '''
    Drone benchmark, with a 6D state space and a 3D control input space.
    '''

    def __init__(self, args):
        DroneDynamics_battery.__init__(self, args, dim=3)

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
        self.uMin = [-1, -1, -1]
        self.uMax = [1, 1, 1]
        self.num_actions = [5, 5, 5]

        v_min = -3.5 # -3.5 not enough (given 0.50 satprob)
        v_max = 3.5

        self.max_charge = 100

        # Expand the battery_charge boundary by 5 to ensure we do not go out of bounds in the simulation
        self.partition['boundary'] = np.array([[-7, v_min, -7, v_min, -7, v_min, 0], [7, v_max, 7, v_max, 7, v_max, self.max_charge+5]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([28, 14, 28, 14, 28, 14, 21])
        
        self.goal = np.array([
            [[5, v_min, 5, v_min, -7, v_min, 50], [7, v_max, 7, v_max, 7, v_max, self.max_charge]]
        ], dtype=float)

        self.critical = np.array([
            # [[-7, v_min, 1, v_min, -7, v_min, 0], [-1, v_max, 3, v_max, 7, v_max, self.max_charge]],
            # [[3, v_min, -7, v_min, -7, v_min, 0], [7, v_max, -3, v_max, 7, v_max, self.max_charge]],
        ], dtype=float)

        self.charging_station = np.array([
            [[-7, v_min, -2, v_min, -2, v_min, 0], [-3, v_max, 2, v_max, 2, v_max, self.max_charge]]
        ], dtype=float)

        self.x0 = np.array([-5, 0.01, -5, 0.01, 0.01, 0.01, 50])

        self.pi_arch = [64, 64]
        self.vf_arch = [64, 64]
        self.inflation_rate = [(-2, 2), (-1, 1), (-2, 2), (-1, 1), (-2, 2), (-1, 1), (-1, 1)]

        return