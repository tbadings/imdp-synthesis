from functools import partial
from benchmarks.models import Test1DDynamics
import jax
import jax.numpy as jnp
import numpy as np
from benchmarks.dynamics import setmath
from core.rl.config import RLConfig


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
            inflation_rate=[(-1, 1)],
        )

        return
