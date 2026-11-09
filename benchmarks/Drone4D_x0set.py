from functools import partial
from benchmarks.models import DroneDynamics, DroneDynamics_battery
import jax
import jax.numpy as jnp
import numpy as np
import scipy 
from benchmarks.dynamics import setmath
from core.rl.config import RLConfig


class Drone4D_x0set(DroneDynamics):
    '''
    Drone benchmark, with a 4D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DroneDynamics.__init__(self, args)

        self.plot_dimensions = [0, 2]

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 1

        self.set_spec(args)

    def set_spec(self, args):
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

        cell_width = (self.partition['boundary'][1] - self.partition['boundary'][0]) / self.partition['number_per_dim']

        self.x0 = np.array([-15.0, 0.00, -15.0, 0.00]) + 0.5 * cell_width

        # The initial state set is defined through the provided model version
        factor = 0.5 + np.array([1 * args.model_version, 2, 1 * args.model_version, 2])
        self.x0_box = np.array([self.x0 - factor * cell_width,
                                self.x0 + factor * cell_width])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="ppo",
            # TODO: Long training is still needed here; can we reduce that?
            total_timesteps=3000000,
            RL_actions_per_state=9,
            inflation_rate=[(-4, 4), (-2, 2), (-4, 4), (-2, 2)],
            proximity_dims = [0, 2],
            proximity_penalty=0.5,
            per_step_cost=0.05,
        )

        return