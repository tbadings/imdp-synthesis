'''
This is the main Python file for DynAbs-JAX.
The file can be run from the terminal as

```Python3 RunFile.py --model <model-name> ...```

For all available arguments, please see the function :func:`core.options.parse_arguments`.
'''

import datetime
import logging
import os
import time
from pathlib import Path
import jax
import numpy as np

import benchmarks
from core.abstraction.probability_intervals import compute_probability_intervals
from core.abstraction.forward_reachability import RectangularForward
from core.options import parse_arguments
from core.abstraction.partition import RectangularPartition
from core.abstraction.imdp import IMDP
from core.abstraction.rvi_jax import RVI_JAX

class _CleanConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f'ERROR: {message}'
        if record.levelno >= logging.WARNING:
            return f'WARN: {message}'
        if record.levelno == logging.DEBUG:
            return f'DEBUG: {message}'
        return message


def configure_logging(log_level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_CleanConsoleFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))
    logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

if __name__ == '__main__':
    args = parse_arguments()
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    jax.config.update("jax_default_matmul_precision", "high")
    args.floatprecision = np.float32

    if args.gpu:
        jax.config.update('jax_platform_name', 'gpu')
        logger.info('Requested to run on GPU')
    else:
        jax.config.update('jax_platform_name', 'cpu')
        logger.info('Requested to run on CPU')

    if args.gpu_rvi:
        args.rvi_device = jax.devices('gpu')[0]
        logger.info('Requested to run RVI on GPU')
    else:
        args.rvi_device = jax.devices('cpu')[0]
        logger.info('Requested to run RVI on CPU')

    logger.info('JAX backend in use: %s', args.rvi_device.platform)
    logger.debug('JAX devices (%s): %s', args.rvi_device.platform, jax.devices(args.rvi_device.platform))

    np.random.seed(args.seed)
    args.jax_key = jax.random.PRNGKey(args.seed)

    # In debug mode, configure jax to use Float64 (for more accurate computations)
    if args.debug:
        from jax import config

        config.update("jax_enable_x64", True)

    # Set current working directory
    args.cwd = os.path.dirname(os.path.abspath(__file__))
    args.root_dir = Path(args.cwd)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.info('Run %s | model=%s | noise=%s | batch=%d', stamp, args.model, args.noise_distr, args.batch_size)
    logger.debug('Arguments: %s', vars(args))

    # Define and parse model
    model = benchmarks.create_model(args)

    t = time.time()

    # Create partition of the continuous state space into convex polytope
    partition = RectangularPartition(model=model)
    
    # Create actions based on forward reachable sets
    actions = RectangularForward(args=args, partition=partition, model=model)
    actions_inputs = actions.id_to_input
    
    P_full, S_id, A_id, P_absorbing = compute_probability_intervals(args=args, 
                                                                    model=model, 
                                                                    partition=partition, 
                                                                    actions=actions,
                                                                    vectorized=True)

    # assert False
    del actions

    imdp = IMDP(partition=partition,
                states=np.array(partition.regions['idxs']),
                actions_inputs=actions_inputs,
                x0=model.x0,
                goal_regions=np.array(partition.goal['bools']),
                critical_regions=np.array(partition.critical['bools']),
                P_full=P_full,
                S_id=S_id,
                A_id=A_id,
                P_absorbing=P_absorbing)

    logger.info('Generating abstraction took %.3f sec.', (time.time() - t))

    # %% Run dynamic programming to compute optimal policy

    with jax.default_device(args.rvi_device):
        logger.info('Computing optimal policy via robust dynamic programming...')
        t = time.time()
        V, policy, policy_inputs = RVI_JAX(
            args=args, 
            imdp=imdp, 
            s0=partition.x2state(model.x0)[0], 
            max_iterations=10000, 
            epsilon=1e-6, 
            RND_SWEEPS=True, 
            BATCH_SIZE=10000, 
            policy_iteration=args.policy_iteration)
        logger.info('RVI with JAX (random-batched asynchronous) took %.3f sec.', (time.time() - t))

    # %% Simulations and plot

    sim_policy = policy
    sim_policy_inputs = policy_inputs
    sim_values = V

    from core.validate.simulate import MonteCarloSim
    from core.plotting.traces import plot_traces
    from core.plotting.heatmap import heatmap

    sim = MonteCarloSim(model, partition, sim_policy, sim_policy_inputs, model.x0, verbose=False, iterations=1000)
    logger.info('Empirical satisfaction probability: %s', sim.results['satprob'])

    plot_traces(args, stamp, model.plot_dimensions, partition, model, sim.results['traces'], line=False, num_traces=10, add_unsafe_box=False,)
    heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=sim_values, filename="heatmap_satprob")
    heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=sim_policy_inputs[:,0], filename="heatmap_inputs")
    
    if args.model == 'Pendulum':
        model.plot_trajectory_gif(np.array(sim.results['traces'][0]['x'])[:,0], filename=f'output/pendulum_{stamp}.gif')

    if args.model == 'MountainCar':
        model.plot_trajectory_gif(np.array(sim.results['traces'][0]['x'])[:,0], filename=f'output/mountaincar_{stamp}.gif')
        
# %%
