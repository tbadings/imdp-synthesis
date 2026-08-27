import argparse
import warnings


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError('Expected a non-negative integer.')
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('Expected a positive integer.')
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError('Expected a value in [0, 1].')
    return parsed


def parse_arguments(argv=None):
    '''
    Function to parse arguments provided

    :return: Object with all arguments
    '''

    # Options
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, perform additional checks to debug python")
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for random number generators (Jax, Numpy)")
    parser.add_argument('--decimals', type=_nonnegative_int, default=4,
                        help="Number of decimals to work with for storing probabilities")
    parser.add_argument('--pAbs_min', type=_probability, default=0.0001,
                        help="Minimum probability for absorbing states")
    parser.add_argument('--satprob', type=float, default=1,
                        help="Lower bound on the satisfaction probability to synthesize a policy for (if <1, then the policy synthesis is terminated once the bound is met).")

    parser.add_argument('--model', type=str, default='',
                        help="Benchmark model to run")
    parser.add_argument('--model_version', type=int, default=0,
                        help="Version of the model to use (optinal; 0 by default)")
    parser.add_argument('--noise_distr', type=str, default='gaussian', choices=['gaussian', 'normal', 'triangular'], # 'normal' is alias for 'gaussian'
                        help="Noise distribution type to use ('normal' is treated as 'gaussian')")
    
    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=False,
                        help="If true, run on GPU. Otherwise, run on CPU")
    parser.add_argument('--gpu_rvi', action=argparse.BooleanOptionalAction, default=False,
                        help="If true, run RVI on GPU. Otherwise, run on CPU")
    
    parser.add_argument('--policy_iteration', action=argparse.BooleanOptionalAction, default=True,
                        help="If true, run policy iteration. Otherwise, run value iteration")
    parser.add_argument('--solver', type=str, default='jax', choices=['jax', 'storm'],
                        help="Solver backend to use for robust dynamic programming")

    parser.add_argument('--mode', type=str, default='fori_loop',
                        help="Should be one of 'fori_loop', 'vmap', 'python'")
    parser.add_argument('--batch_size', type=_positive_int, default=100,
                        help="For computing the transition probability intervals, the number of states to process in a vectorized fashion (Warning: increasing this too much drastically increases memory usage for JIT compilation by JAX!)")
    parser.add_argument('--frs_batch_size', type=_positive_int, default=1000,
                        help="Number of state regions to process per batch when computing forward reachable sets. Larger values reduce Python-JAX round trips but increase peak memory usage.")
    parser.add_argument('--shrink_frs', type=float, default=0.0001,
                        help="Amount to shrink forward reachable set bounds inward for numerical stability (avoids misclassification when the FRS lands exactly on a cell boundary).")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity level')
    parser.add_argument('--output_root', type=str, default='output',
                        help='Base directory where per-run output folders are created')
    parser.add_argument('--load_checkpoint', type=str, default=None, metavar='PATH',
                        help='Path to a checkpoint.pkl file saved by a previous run. '
                             'When set, the IMDP abstraction generation is skipped and '
                             'model, partition, and IMDP are loaded from the checkpoint.')
    parser.add_argument('--save_checkpoint', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, save checkpoints during execution")
    # Plotting options
    parser.add_argument('--plot_grid', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plot unit grids in figures")
    parser.add_argument('--plot_title', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plot titles in figures")
    parser.add_argument('--plot_ticks', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, plot ticks in figures")

    # Reinforcement learning options.
    # Every option here defaults to None, meaning "not given". The actual defaults live in
    # `core.rl.config.RLConfig`, each benchmark overrides the ones it cares about through its
    # `rl_config` attribute, and anything passed here overrides both. See `resolve_rl_config`.
    rl = parser.add_argument_group(
        "reinforcement learning",
        "Overrides for the benchmark's rl_config (core.rl.config.RLConfig holds the defaults).",
    )

    # Rollouts
    rl.add_argument("--total_timesteps", type=_positive_int, default=None,
                    help="Total number of environment steps to train PPO for.")
    rl.add_argument("--eval_episodes", type=_positive_int, default=None,
                    help="Number of evaluation rollouts used to measure the policy and seed the tube.")
    rl.add_argument("--max_steps", type=_positive_int, default=None,
                    help="Truncation for *training* episodes. Shorter episodes reset more often, "
                         "which spreads the training data over the whole state space.")
    rl.add_argument("--eval_steps", type=_positive_int, default=None,
                    help="Rollout horizon for evaluation and tube construction. Defaults to --max_steps; "
                         "set it when --max_steps is shortened for training, so the rollouts still have "
                         "room to reach the goal.")

    # Reward function
    rl.add_argument("--goal_reward", type=float, default=None,
                    help="Reward for reaching the goal set (terminal).")
    rl.add_argument("--unsafe_penalty", type=float, default=None,
                    help="Reward for entering a critical region (terminal); should be negative.")
    rl.add_argument("--out_of_bounds_penalty", type=float, default=None,
                    help="Reward for leaving the state-space boundary (terminal); should be negative.")
    rl.add_argument("--distance_cost", type=float, nargs='+', default=None,
                    help="Gain on the Euclidean distance-to-goal cost applied on non-terminal steps: the reward "
                         "loses distance_cost * d, where d is the normalised distance from the state to the "
                         "goal set (0 inside it, 1 at the far corner). Either a single value, which scales "
                         "every state dimension, or one value per state dimension, e.g. --distance_cost 0.05 0.0 "
                         "to count position but not velocity. 0 disables it. Note that the terminal penalties "
                         "must stay larger than the cost of surviving without reaching the goal, "
                         "(per_step_cost + norm(distance_cost)) / (1 - gamma), otherwise ending the episode on "
                         "purpose becomes optimal.")
    rl.add_argument("--per_step_cost", type=float, default=None,
                    help="Cost subtracted from the reward on every non-terminal step. 0 disables it; x imposes a cost of x per step.")

    # PPO hyperparameters
    rl.add_argument("--ent_coef", type=float, default=None,
                    help="PPO entropy coefficient. Increase (e.g. 0.005) to encourage exploration.")
    rl.add_argument("--learning_rate", type=float, default=None,
                    help="Adam learning rate for PPO.")
    rl.add_argument("--rl_batch_size", type=_positive_int, default=None,
                    help="Batch size for PPO updates across minibatches")
    rl.add_argument("--n_steps", type=_positive_int, default=None,
                    help="Number of steps to run for each environment per update in PPO")
    rl.add_argument("--n_envs", type=_positive_int, default=None,
                    help="Number of parallel vectorized environments in JAX (increase to use more CPU cores)")
    rl.add_argument("--subproc", action=argparse.BooleanOptionalAction, default=None)
    rl.add_argument("--finetune_steps", type=_nonnegative_int, default=None,
                    help="Number of training steps for fine-tuning the policy to stay within active states. Set to 0 to skip fine-tuning.")
    rl.add_argument("--pi_arch", type=_positive_int, nargs='+', default=None,
                    help="Hidden layer sizes for the policy (actor) network, e.g. --pi_arch 128 128.")
    rl.add_argument("--vf_arch", type=_positive_int, nargs='+', default=None,
                    help="Hidden layer sizes for the value function (critic) network, e.g. --vf_arch 256 256 256.")
    rl.add_argument("--update_epochs", type=_positive_int, default=None,
                    help="Number of PPO optimization epochs per rollout update batch.")
    rl.add_argument("--clip_eps", type=float, default=None,
                    help="PPO clipping parameter epsilon for policy and value function loss clipping.")
    rl.add_argument("--vf_coef", type=float, default=None,
                    help="Coefficient for value function loss in total PPO loss.")
    rl.add_argument("--max_grad_norm", type=float, default=None,
                    help="Maximum gradient norm for gradient clipping in PPO.")
    rl.add_argument("--gamma", type=float, default=None,
                    help="Discount factor for Generalized Advantage Estimation (GAE).")
    rl.add_argument("--gae_lambda", type=float, default=None,
                    help="Lambda parameter for Generalized Advantage Estimation (GAE).")
    rl.add_argument("--adam_eps", type=float, default=None,
                    help="Epsilon parameter for Adam optimizer.")

    # Tube around the RL rollouts
    rl.add_argument("--RL_actions_per_state", type=_positive_int, default=None,
                    help="The number of active actions to keep per state based on the RL policy's action preferences. "
                         "This is used to create a sparse abstraction focused on the most relevant actions for each state.")
    rl.add_argument("--tube_method", type=str, default=None, choices=["inflation", "smart"],
                    help="Method for creating the state-space tube around the policy's trajectories used for abstraction.")
    rl.add_argument("--smart_tube_rate", type=float, default=None,
                    help="Noise support rate used for reachability-guided (smart) tube expansion.")

    # Parse arguments
    args = parser.parse_args(argv)

    # Canonicalize alias.
    if args.noise_distr == 'normal':
        args.noise_distr = 'gaussian'

    if args.solver == 'storm' and args.policy_iteration:
        warnings.warn(
            "solver='storm' does not support policy_iteration; Storm runs value iteration instead.",
            UserWarning,
            stacklevel=2,
        )

    return args
