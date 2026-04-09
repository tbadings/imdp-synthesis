import logging
from tqdm import tqdm
import time
import stormpy
import numpy as np

class BuilderStorm:
    """
    Class to construct the IMDP abstraction and compute an optimal Markov policy using Storm.
    """

    def __init__(self, IMDP):
        '''
        Generate the IMDP abstraction
        '''

        self.builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                                           has_custom_row_grouping=True, row_groups=0)

        

        # Predefine the pycarl intervals
        self.intervals_raw = {}
        self.intervals_raw[(1, 1)] = stormpy.pycarl.Interval(1, 1)

        # Reshape all probability intervals
        print('- Generate pycarl intervals...')
        P_full_flat = np.concatenate([np.concatenate(Ps) for Ps in IMDP.P_full if len(Ps) > 0])
        P_absorbing_flat = np.concatenate([Ps for Ps in IMDP.P_absorbing if len(Ps) > 0])

        print('-- Probability intervals reshaped')

        P_stacked = np.vstack((P_full_flat, P_absorbing_flat))
        P_unique = np.unique(P_stacked, axis=0)

        # Experimental: Determining the unique elements is much faster with Pandas, but sometimes leads to rounding errors.
        # P_unique = np.round(np.sort(pd.DataFrame(P_stacked).drop_duplicates(), axis=0), args.decimals)
        # assert np.all(P_unique2 == P_unique)

        print('-- Unique probability intervals determined')

        # Enumerate only over unique probability intervals
        for P in tqdm(P_unique):
            self.intervals_raw[tuple(P)] = stormpy.pycarl.Interval(P[0], P[1])

        self.intervals_state = {}
        self.intervals_absorbing = {}

        print('-- Pycarl intervals created')

        print('\n- Store intervals for individual transitions...')
        for s in tqdm(IMDP.states):
            self.intervals_state[s] = {}
            self.intervals_absorbing[s] = {}
            for i, a in enumerate(IMDP.P_id[s].keys()):
                self.intervals_state[s][a] = {}

                # Add intervals for each successor state
                for s_next, prob in zip(IMDP.P_id[s][a], IMDP.P_full[s][i]):
                    self.intervals_state[s][a][s_next] = self.intervals_raw[tuple(prob)]

                # Add intervals for other states
                if IMDP.P_absorbing[s][i][1] > 0:
                    self.intervals_absorbing[s][a] = self.intervals_raw[tuple(IMDP.P_absorbing[s][i])]
        row = 0

        # Total number of choices = sum of choices in all states (always >=1), and always a single choice in a goal/critical state.
        # Add one for the absorbing state.
        total_choices = np.sum([max(1, len(p.keys())) if s not in IMDP.goal_regions and s not in IMDP.critical_regions else 1 for s, p in IMDP.P_id.items()]) + 1
        choice_labeling = stormpy.storage.ChoiceLabeling(total_choices)
        choice_labels = {str(i) for i in range(-1, len(IMDP.actions_inputs))}
        for label in choice_labels:
            choice_labeling.add_label(label)
        
        # For all states
        print('\n- Build iMDP...')
        for s in tqdm(IMDP.states):

            # if s == s0:
                # print(f'- Current state is initial state (s={s})')

            # For each state, create a new row group
            self.builder.new_row_group(row)
            enabled_in_s = IMDP.P_id[s].keys()

            # if s == s0:
            #     print(f'- Number of actions enabled: {len(enabled_in_s)}')

            # If no actions are enabled at all, add a deterministic transition to the absorbing state
            if len(enabled_in_s) == 0 or s in IMDP.critical_regions:
                choice_labeling.add_label_to_choice(str(-1), row)
                self.builder.add_next_value(row, IMDP.absorbing_state, self.intervals_raw[(1, 1)])
                row += 1

            elif s in IMDP.goal_regions:
                choice_labeling.add_label_to_choice(str(-1), row)
                self.builder.add_next_value(row, s, self.intervals_raw[(1, 1)])
                row += 1

            else:
                # For every enabled action
                for a in enabled_in_s:
                    choice_labeling.add_label_to_choice(str(a), row)

                    # if s == s0:
                    #     print(f'-- Actions {a}: ',self.intervals_state[s][a])

                    for s_next, intv in self.intervals_state[s][a].items():
                        self.builder.add_next_value(row, s_next, intv)

                    # Add transitions to absorbing state
                    if a in self.intervals_absorbing[s]:
                        self.builder.add_next_value(row, IMDP.absorbing_state, self.intervals_absorbing[s][a])

                    row += 1

        for s in [IMDP.absorbing_state]:
            self.builder.new_row_group(row)
            self.builder.add_next_value(row, s, self.intervals_raw[(1, 1)])
            choice_labeling.add_label_to_choice(str(-1), row)
            row += 1

        self.nr_states = IMDP.nr_states

        matrix = self.builder.build()
        logging.debug(matrix)

        # Create state labeling
        state_labeling = stormpy.storage.StateLabeling(self.nr_states)

        # Define initial states
        state_labeling.add_label('init')
        state_labeling.add_label_to_state('init', IMDP.s_init)

        # Add absorbing (unsafe) states
        state_labeling.add_label('absorbing')
        state_labeling.add_label_to_state('absorbing', IMDP.absorbing_state)

        # Add critical (unsafe) states
        state_labeling.add_label('critical')
        for s in IMDP.critical_regions:
            state_labeling.add_label_to_state('critical', s)

        # Add goal states
        state_labeling.add_label('goal')
        for s in IMDP.goal_regions:
            state_labeling.add_label_to_state('goal', s)

        components = stormpy.SparseIntervalModelComponents(transition_matrix=matrix, state_labeling=state_labeling)
        components.choice_labeling = choice_labeling
        self.imdp = stormpy.storage.SparseIntervalMdp(components)

    def compute_reach_avoid(self, maximizing=True):
        '''
        Compute a Markov policy that maximizes the probability of satisfying the reach-avoid property

        :param maximizing: If True, maximise the reachability; Otherwise, minimise.
        '''

        prop = stormpy.parse_properties('P{}=? [F "goal"]'.format('max' if maximizing else 'min'))[0]
        env = stormpy.Environment()
        env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.value_iteration

        # Compute reach-avoid probability
        task = stormpy.CheckTask(prop.raw_formula, only_initial_states=False)
        task.set_produce_schedulers()
        task.set_robust_uncertainty(True)
        self.result_generator = stormpy.check_interval_mdp(self.imdp, task, env)
        self.results = np.array(self.result_generator.get_values())[:self.nr_states]

        return

    def get_policy(self, actions_inputs):
        '''
        Extract optimal policy from the IMDP object

        :param actions_inputs: Object with Action inputs info.
        '''

        assert self.result_generator.has_scheduler
        scheduler = self.result_generator.scheduler
        policy = np.zeros(self.nr_states, dtype=int)
        policy_inputs = np.zeros((self.nr_states, actions_inputs.shape[1]), dtype=float)
        for state in self.imdp.states:
            choice = scheduler.get_choice(state)
            action_index = choice.get_deterministic_choice()
            action = state.actions[action_index]

            action_label = int(list(action.labels)[0])
            policy[int(state)] = action_label
            policy_inputs[int(state)] = actions_inputs[action_label]

        return policy, policy_inputs

    def get_label(self, s):
        '''
        Get label for given state.

        :param s: State to get label for.
        :return: Label.
        '''

        label = ''
        if 'goal' in self.imdp.states[s].labels:
            label += 'goal,'
        elif 'absorbing' in self.imdp.states[s].labels:
            label += 'absorbing,'
        elif 'critical' in self.imdp.states[s].labels:
            label += 'critical,'

        return label

    def print_transitions(self, state, action, actions_inputs, partition):
        '''
        Print the transitions for the given state.

        :param state: State to print transitions for.
        :param action: Action to print transitions for.
        :param actions: Object with action info.
        :param partition: Object with partition info.
        '''

        if type(state) in [list, tuple]:
            state = int(partition.region_idx_array[tuple(state)])

        print('\n----------')

        print('From state {} at position {} (label: {}), with action {} with inputs {}:'.format(state, partition.region_idx_inv[state], self.get_label(state), action,
                                                                                                actions_inputs[action]))
        print(' - Optimal value: {}'.format(self.results[state]))
        print(' ---')
        print(' - State lower bound: {}'.format(partition.regions['lower_bounds'][state]))
        print(' - State upper bound: {}'.format(partition.regions['upper_bounds'][state]))
        print(' ---')
        print(' - Action FRS lower bound: {}'.format(actions.frs[state]['lb'][action]))
        print(' - Action FRS upper bound: {}'.format(actions.frs[state]['ub'][action]))
        print(' ---')

        try:
            SA = self.imdp.states[state].actions[action]

            for transition in SA.transitions:
                s_prime = transition.column
                if s_prime < len(partition.region_idx_inv):
                    idx = partition.region_idx_inv[s_prime]
                else:
                    idx = '<out-partition>'
                print(" --- With probability {}, go to state {} at position {} (label: {})".format(transition.value(), s_prime, idx, self.get_label(s_prime)))

        except:
            print(' - Error: This action does not exist in this state')

        print('----------')

    def get_value_from_tuple(self, x, partition):
        '''
        Get the reach-avoid probability for a given state

        :param x: State to return reach-avoid probability for.
        :param partition: Object with partition info.
        :return: Value in the given state.
        '''

        s = partition.x2state(x)
        return self.results[s]