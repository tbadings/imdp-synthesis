from functools import partial
from benchmarks.models import DubinsDynamics4D
import jax
import jax.numpy as jnp
import numpy as np
from benchmarks.dynamics import setmath
from core.rl.config import RLConfig


class Dubins4D(DubinsDynamics4D):
    '''
    Dubin's vehicle benchmark, with a 4D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DubinsDynamics4D.__init__(self, args)

        self.plot_dimensions = [0, 1]

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
        self.uMin = [-0.5, -0.5]
        self.uMax = [0.5, 0.5]
        self.num_actions = [5, 5]

        v_min = self.v_min
        v_max = self.v_max

        self.partition['boundary'] = np.array([[-10, 0, -np.pi, v_min], [10, 10, np.pi, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([100, 50, 100, 100])

        self.goal = np.array([
            [[6, 6, -np.pi, v_min], [9, 9, np.pi, v_max]]
        ], dtype=float)

        self.critical = np.array([
            [[4, 5, -2 * np.pi, v_min], [5, 10, 2 * np.pi, v_max]],
            [[-1, 0, -2 * np.pi, v_min], [0, 5, 2 * np.pi, v_max]],
            [[-5, 4, -2 * np.pi, v_min], [-1, 5, 2 * np.pi, v_max]],
        ], dtype=float)

        self.x0 = np.array([-3, 2, np.pi/2, 0.0])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            pi_arch=[256, 256], 
            vf_arch=[256, 256],
            total_timesteps=2000000,
            proximity_dims=[0, 1],
            proximity_penalty=0.1,
            RL_actions_per_state=9,
            inflation_rate=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)],
        )

        return
