from functools import partial
from benchmarks.models import DroneDynamics_2agent
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from benchmarks.dynamics import setmath
from core.rl.config import RLConfig



class Drone4D_2agent(DroneDynamics_2agent):
    '''
    Two independent Drone4D agents, with an 8D state space and a 4D control input space.
    '''

    def __init__(self, args):
        DroneDynamics_2agent.__init__(self, args)

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
        self.uMin = [-1, -1, -1, -1]
        self.uMax = [1 ,1, 1, 1]
        self.num_actions = [5, 5, 5, 5]

        v_min = self.v_min
        v_max = self.v_max

        self.partition['boundary'] = np.array([[-2.5, v_min, -2.5, v_min, -2.5, v_min, -2.5, v_min],
                                               [2.5, v_max, 2.5, v_max, 2.5, v_max, 2.5, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([10, 10, 10, 10, 10, 10, 10, 10])

        self.goal = np.array([
            [[1, v_min, 1, v_min, 1, v_min, -2.5, v_min],
             [2.5, v_max, 2.5, v_max, 2.5, v_max, -1, v_max]]
        ], dtype=float)

        self.critical = np.array([
        ], dtype=float)

        self.x0 = np.array([-2.25, 0.01, -2.25, 0.01, -2.25, 0.01, 2.25, 0.01])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="ppo",
            # TODO: Long training is still needed here; can we reduce that?
            total_timesteps=1000000,
            RL_actions_per_state=3**4,
            inflation_rate=[(-2, 2), (-1, 1), (-2, 2), (-1, 1), (-2, 2), (-1, 1), (-2, 2), (-1, 1)],
                        # [(-4, 4), (-2, 2), (-4, 4), (-2, 2), (-4, 4), (-2, 2), (-4, 4), (-2, 2)],
            proximity_dims = [0, 2, 4, 6],
            goal_reward=50.0,
            unsafe_penalty=-50.0,
            out_of_bounds_penalty=-50.0,
            distance_cost=[0.1, 0.0, 0.1, 0.0,
                        0.1, 0.0, 0.1, 0.0],
            per_step_cost=0.01,
            proximity_penalty=0.1,
            eval_episodes=100,
            pi_arch=[256, 256],
            vf_arch=[256, 256],
        )

        return
