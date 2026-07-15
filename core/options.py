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
    parser.add_argument('--RL_actions_per_state', type=_positive_int, default=4,
                        help="The number of active actions to keep per state based on the RL policy's action preferences. This is used to create a sparse abstraction focused on the most relevant actions for each state.")

    # Plotting options
    parser.add_argument('--plot_grid', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plot unit grids in figures")
    parser.add_argument('--plot_title', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plot titles in figures")
    parser.add_argument('--plot_ticks', action=argparse.BooleanOptionalAction, default=True,
                        help="If True, plot ticks in figures")

    parser.add_argument("--total_timesteps", type=int, default=200000)
    parser.add_argument("--eval_episodes", type=int, default=2500)
    parser.add_argument("--max_steps", type=int, default=32)

    parser.add_argument("--goal_reward", type=float, default=100.0)
    parser.add_argument("--unsafe_penalty", type=float, default=-100.0)
    parser.add_argument("--out_of_bounds_penalty", type=float, default=-100.0)
    parser.add_argument("--revisit_penalty", type=float, default=0.05)
    parser.add_argument("--progress_reward", action=argparse.BooleanOptionalAction, default=False,
                        help="If True, use dense distance-to-goal shaping (progress reward minus a per-step penalty) for non-terminal steps instead of a flat 0.")

    parser.add_argument("--ent_coef", type=float, default=0.0,
                        help="PPO entropy coefficient. SB3 default is 0.0; increase (e.g. 0.005) to encourage exploration ")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--rl_batch_size", type=int, default=256, help="Batch size for PPO")
    parser.add_argument("--n_steps", type=int, default=128, help="Number of steps to run for each environment per update in PPO")
    parser.add_argument("--n_envs", type=int, default=32)
    parser.add_argument("--subproc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--finetune_steps", type=int, default=0,
                        help="Number of training steps for fine-tuning the policy to stay within active states. Set to 0 to skip fine-tuning.")
    parser.add_argument("--pi_arch", type=int, nargs='+', default=None,
                        help="Hidden layer sizes for the policy (actor) network, e.g. --pi_arch 128 128. Defaults to the benchmark's built-in value.")
    parser.add_argument("--vf_arch", type=int, nargs='+', default=None,
                        help="Hidden layer sizes for the value function (critic) network, e.g. --vf_arch 256 256 256. Defaults to the benchmark's built-in value.")

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
