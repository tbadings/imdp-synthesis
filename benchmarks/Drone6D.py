from functools import partial
from benchmarks.models import DroneDynamics, DroneDynamics_battery
import jax
import jax.numpy as jnp
import numpy as np
import scipy 
from benchmarks.dynamics import setmath
from core.rl.config import RLConfig


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

        v_min = self.v_min
        v_max = self.v_max

        self.partition['boundary'] = np.array([[-17, v_min, -9, v_min, -7, v_min], 
                                               [17, v_max, 9, v_max, 7, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([68, 8, 36, 8, 28, 8])

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

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="sac",
            total_timesteps=10000000,
            RL_actions_per_state=27,
            proximity_dims=[0, 2, 4],
            proximity_penalty=0.5,
            per_step_cost=0.05,
            inflation_rate=[(-3, 3), (-2, 2), (-3, 3), (-2, 2), (-3, 3), (-2, 2)],
        )
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

        v_min = self.v_min
        v_max = self.v_max

        self.partition['boundary'] = np.array([[-7, v_min, -7, v_min, -7, v_min], [7, v_max, 7, v_max, 7, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([28, 10, 28, 10, 28, 10])

        self.goal = np.array([
            [[3, v_min, 3, v_min, -7, v_min], [7, v_max, 7, v_max, 7, v_max]]
        ], dtype=float)

        self.critical = np.array([
            [[-7, v_min, 1, v_min, -7, v_min], [-1, v_max, 3, v_max, 7, v_max]],
            [[3, v_min, -7, v_min, -7, v_min], [7, v_max, -3, v_max, 7, v_max]],
        ], dtype=float)

        self.x0 = np.array([-5.5, 0.01, -5.5, 0.01, 0.01, 0.01])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            pi_arch=[32, 32],
            vf_arch=[32, 32],
            total_timesteps=100000,
            eval_episodes=1000,
            goal_reward=5,
            unsafe_penalty=-5,
            out_of_bounds_penalty=-5,
            per_step_cost=0.1,
            distance_cost=0.0,
            RL_actions_per_state=27,
            inflation_rate=[(-2, 2), (-1, 1), (-2, 2), (-1, 1), (-2, 2), (-1, 1)],
        )

        return


class Drone6D_battery(DroneDynamics_battery):
    '''
    Drone benchmark with 6D motion, one battery state, and 3D control.
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

        v_min = self.v_min
        v_max = self.v_max

        self.max_charge = 100

        # The x and y axes span the same bounds as Drone4D_battery. The z axis
        # spans [-2, 2] at the same 0.5 m resolution as x and y.
        self.partition['boundary'] = np.array([
            [-10, v_min, -10, v_min, -2, v_min, 0],
            [10, v_max, 10, v_max, 2, v_max, self.max_charge],
        ])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([40, 10, 40, 10, 8, 10, 40])

        self.goal = np.array([
            [
                [6, v_min, 6, v_min, -2, v_min, 20],
                [10, v_max, 10, v_max, 2, v_max, self.max_charge],
            ]
        ], dtype=float)

        self.critical = np.empty((0, 2, 7), dtype=float)

        self.charging_station = np.array([
            [
                [-9, v_min, -2, v_min, -2, v_min, 0],
                [-5, v_max, 2, v_max, 2, v_max, self.max_charge],
            ]
        ], dtype=float)

        self.x0 = np.array([-5, 0.01, -9, 0.01, 0, 0.01, 50])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="ppo",
            total_timesteps=1000000,
            pi_arch=[256, 256],
            vf_arch=[256, 256],
            RL_actions_per_state=27,
            proximity_dims=[0, 2, 4],
            proximity_penalty=0.5,
            per_step_cost=0.05,
            inflation_rate=[
                (-3, 3),
                (-2, 2),
                (-3, 3),
                (-2, 2),
                (-3, 3),
                (-2, 2),
                (-3, 3),
            ],
        )

        return
