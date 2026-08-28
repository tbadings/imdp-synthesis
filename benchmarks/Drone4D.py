from functools import partial
from benchmarks.models import DroneDynamics, DroneDynamics_battery
import jax
import jax.numpy as jnp
import numpy as np
import scipy 
from benchmarks.dynamics import setmath
from core.rl.config import RLConfig


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

        v_min = self.v_min
        v_max = self.v_max

        self.partition['boundary'] = np.array([[-20.0, v_min, -20.0, v_min], [20.0, v_max, 20.0, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([80, 10, 80, 10])

        self.goal = np.array([
            [[12.0, v_min, 12.0, v_min], [18.0, v_max, 18.0, v_max]]
        ], dtype=float)

        self.critical = np.array([
            # Lower wall with right-side opening
            [[-20.0, v_min, -7.0, v_min], [8.0, v_max, -4.0, v_max]],
            # Upper wall with left-side opening
            [[-8.0, v_min, 4.0, v_min], [20.0, v_max, 7.0, v_max]],
            # Lower lane obstacle
            [[-4.0, v_min, -14.0, v_min], [4.0, v_max, -10.0, v_max]],
            # Upper lane obstacle 
            [[-4.0, v_min, 10.0, v_min], [4.0, v_max, 14.0, v_max]],
        ], dtype=float)

        self.x0 = np.array([-15.0, 0.01, -15.0, 0.01])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            pi_arch=[256, 256],
            vf_arch=[256, 256],
            # TODO: Long training is still needed here; can we reduce that?
            total_timesteps=2000000,
            # We need short training rollouts but long evaluation rollouts to reach the goal.
            max_steps=32,
            eval_steps=200,
            eval_episodes=5000,
            RL_actions_per_state=9,
            inflation_rate=[(-2, 2), (-1, 1), (-2, 2), (-1, 1)],
            proximity_penalty=1.0,
            proximity_dims = [0, 2],
        )

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

        v_min = self.v_min
        v_max = self.v_max

        self.max_charge = 100

        # Expand the battery_charge boundary by 5 to ensure we do not go out of bounds in the simulation
        self.partition['boundary'] = np.array([[-10, v_min, -10, v_min, 0], [10, v_max, 10, v_max, self.max_charge+5]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([40, 14, 40, 14, 21])
        
        self.goal = np.array([
            [[6, v_min, 6, v_min, 50], [10, v_max, 10, v_max, self.max_charge]]
        ], dtype=float)

        self.critical = np.array([
            # [[-7, v_min, 1, v_min, 0], [-1, v_max, 3, v_max, self.max_charge]],
            # [[3, v_min, -7, v_min, 0], [7, v_max, -3, v_max, self.max_charge]],
        ], dtype=float)

        self.charging_station = np.array([
            [[-9, v_min, -2, v_min, 0], [-5, v_max, 2, v_max, self.max_charge]]
        ], dtype=float)

        self.x0 = np.array([-5, 0.01, -9, 0.01, 50])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            pi_arch=[64, 64],
            vf_arch=[64, 64],
            goal_reward=5,
            unsafe_penalty=-5,
            out_of_bounds_penalty=-5,
            per_step_cost=0.1,
            distance_cost=0.0,
            inflation_rate=[(-2, 2), (-1, 1), (-2, 2), (-1, 1), (-1, 1)],
        )

        return

