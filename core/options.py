import argparse


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

    parser.add_argument('--model', type=str, default='Drone2D',
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

    parser.add_argument('--mode', type=str, default='fori_loop',
                        help="Should be one of 'fori_loop', 'vmap', 'python'")
    parser.add_argument('--batch_size', type=_positive_int, default=100,
                        help="For computing the transition probability intervals, the number of states to process in a vectorized fashion (Warning: increasing this too much drastically increases memory usage for JIT compilation by JAX!)")
    parser.add_argument('--frs_batch_size', type=_positive_int, default=1000,
                        help="Number of state regions to process per batch when computing forward reachable sets. Larger values reduce Python-JAX round trips but increase peak memory usage.")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity level')

    # Plotting options
    parser.add_argument('--plot_grid', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plot unit grids in figures")
    parser.add_argument('--plot_title', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plot titles in figures")
    parser.add_argument('--plot_ticks', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plot ticks in figures")

    # Parse arguments
    args = parser.parse_args(argv)

    # Canonicalize alias.
    if args.noise_distr == 'normal':
        args.noise_distr = 'gaussian'

    return args
