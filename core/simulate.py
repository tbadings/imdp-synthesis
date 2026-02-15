import numpy as np
from tqdm import tqdm


class MonteCarloSim():
    '''
    Class to run Monte Carlo simulations on the discrete-time stochastic system closed under a fixed Markov policy.
    '''

    def __init__(self, model, partition, policy, policy_inputs, x0, iterations=100, sim_horizon=1000, random_initial_state=False, verbose=True, **kwargs):

        print('\nStarting Monte Carlo simulations...')

        self.verbose = verbose

        self.model = model
        self.partition = partition

        self.policy = policy
        self.policy_inputs = policy_inputs
        self.horizon = sim_horizon
        self.iterations = iterations
        self.random_initial_state = random_initial_state

        # Predefine noise to speed up computations
        self.define_noise_values()

        self.results = {
            'satprob': -1,
            'goal_reached': np.full(self.iterations, False, dtype=bool),
            'traces': {}
        }

        # For each of the monte carlo iterations
        for m in tqdm(range(self.iterations)):
            self.results['traces'][m], self.results['goal_reached'][m] = self._runIteration(x0, m)

        self.results['satprob'] = np.mean(self.results['goal_reached'])

    def define_noise_values(self):
        '''
        Predefine the noise values to speed up computations.
        '''

        # Gaussian noise mode
        self.noise = np.random.multivariate_normal(
            np.zeros(self.model.n), self.model.noise['cov'] ** 2,
            (self.iterations, self.horizon))

    def _runIteration(self, x0, m):
        '''
        Run a Monte Carlo simulation from x0.

        :param x0: Initial continuous state.
        :param m: Simulation number.
        :return:
            - trace: Dictionary containing the state and input at each time step.
            - success: Boolean indicating whether goal was reached.
        '''

        # Initialize variables at start of iteration
        success = False
        trace = {'k': [], 'x': [], 'u': []}
        k = 0

        # Initialize the current simulation
        x = np.zeros((self.horizon + 1, self.model.n))
        x_tuple = np.zeros((self.horizon + 1, self.model.n)).astype(int)
        s = np.zeros(self.horizon + 1).astype(int)
        u = np.zeros((self.horizon, self.model.p))
        a = np.zeros(self.horizon).astype(int)

        # Determine initial state
        if self.random_initial_state:
            s0, _ = self.partition.x2state(x0)
            x[0] = np.random.uniform(
                low=self.partition.regions['lower_bounds'][s0],
                high=self.partition.regions['lower_bounds'][s0])

        else:
            x[0] = x0

        # Add current state, belief, etc. to trace
        trace['k'] += [0]
        trace['x'] += [x[0]]

        ######

        # For each time step in the finite time horizon
        while k <= self.horizon:

            # Determine to which region the state belongs
            s[k], in_partition = self.partition.x2state(x[k])

            if in_partition:
                # Save that state is currently in state s_current
                x_tuple[k] = self.partition.region_idx_inv[s[k]]

            else:
                # Absorbing region reached
                x_tuple[k] = -1

                if self.verbose or True:
                    print(f'- Absorbing state reached at k = {k} (x = {x[k]}), so abort')
                return trace, success

            # If current region is the goal state ...
            if s[k] in self.partition.goal['idxs']:
                # Then abort the current iteration, as we have achieved the goal
                success = True
                if self.verbose:
                    print(f'- Goal state reached (x = {x[k]})')
                return trace, success

            # If current region is in critical states...
            elif s[k] in self.partition.critical['idxs']:
                # Then abort current iteration
                if self.verbose or True:
                    print('- Critical state reached, so abort')
                return trace, success

            # Check if we can still perform another action within the horizon
            elif k >= self.horizon:
                return trace, success

            # Retreive the action from the policy
            if len(self.policy.shape) == 1:
                # If infinite horizon, policy does not have a time index
                a[k] = self.policy[s[k]]
                u[k] = self.policy_inputs[s[k]]
            else:
                # If finite horizon, use action for the current time step k
                a[k] = self.policy[k, s[k]]
                u[k] = self.policy_inputs[k, s[k]]

            # if a[k] == -1:
            #     if self.verbose:
            #         print('No policy known, so abort')
            #     return trace, success

            ###

            # If loop was not aborted, we have a valid action
            if self.verbose:
                print(f'In state {s[k]} (x = {x[k]}), take action {a[k]} (u = {u[k]})')

            x[k + 1] = self.model.step(x[k], u[k], self.noise[m, k])

            # Add current state, belief, etc. to trace
            trace['k'] += [k + 1]
            trace['u'] += [u[k]]
            trace['x'] += [x[k + 1]]

            # Increase iterator variable by one
            k += 1

        ######

        return trace, success
