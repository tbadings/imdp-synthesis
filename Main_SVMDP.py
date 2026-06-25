import datetime
import logging
import os
import random
import time
from pathlib import Path
import jax
import numpy as np
import torch

import benchmarks
from core.abstraction.svmdp.forward_reachability import RectangularForward
from core.abstraction.svmdp.svmdp import SVMDP
from core.abstraction.svmdp.rvi_jax import RVI_SVMDP
from core.options import parse_arguments
from core.abstraction.partition import RectangularPartition, SparsePartition
from core.jax_config import configure_jax
from core.utils import configure_logging, add_file_handler
from core.rl import find_active

if __name__ == '__main__':
    args = parse_arguments()
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    configure_jax(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.jax_key = jax.random.PRNGKey(args.seed)

    # Set current working directory
    args.cwd = os.path.dirname(os.path.abspath(__file__))
    args.root_dir = Path(args.cwd)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_output_dir = args.root_dir / args.output_root / f"{stamp}_{args.model}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = run_output_dir
    add_file_handler(run_output_dir, stamp)
    logger.info('Run %s | model=%s | noise=%s', stamp, args.model, args.noise_distr)
    logger.info('Output directory: %s', run_output_dir)

    logger.info('\n=== Generating SVMDP from scratch ===')
    logger.debug('Arguments: %s', vars(args))

    model = benchmarks.create_model(args)

    t = time.time()

    # Partition the continuous state space.
    partition = RectangularPartition(model=model)

    s_init, s_init_exists = partition.x2state(model.x0)
    if not s_init_exists:
        raise ValueError(f"Initial state x0={model.x0} is not in the partition.")

    # Set up the noise partition (required by RectangularForward for SVMDP).
    # num_cells_per_dim: number of noise cells per dimension; zero-noise dims are
    # forced to 1 cell automatically inside set_partition_probs.
    noise_cells_per_dim = getattr(args, 'noise_partition_cells', None)
    if noise_cells_per_dim is None:
        noise_cells_per_dim = [10] * model.n
    model.noise.set_partition_probs(noise_cells_per_dim)
    logger.info(
        'Noise partition: %d cells total  |  remainder=%.4f',
        len(model.noise.partition['probs']),
        float(model.noise.partition['remainder']),
    )

    # Compute forward reachable sets and noise-shifted successor cell IDs.
    actions = RectangularForward(args=args, partition=partition, model=model)

    # All partition states have all actions enabled (rectangular partition).
    states = np.array(partition.regions['idxs'])
    A_id = {int(s): list(range(actions.num_actions)) for s in states}

    svmdp = SVMDP(
        partition=partition,
        states=states,
        x0=model.x0,
        goal_regions=np.array(partition.goal['bools']),
        critical_regions=np.array(partition.critical['bools']),
        A_id=A_id,
        actions=actions,
        noise_probs=model.noise.partition['probs'],
        noise_remainder=model.noise.partition['remainder'],
    )

    logger.info('Initial state x0=%s → state index %d', model.x0, s_init)
    logger.info('Generating SVMDP abstraction took %.3f sec.', time.time() - t)

    # %% Run value iteration on the SVMDP

    logger.info('\n=== Computing optimal policy via SVMDP value iteration ===')
    t = time.time()
    with jax.default_device(args.rvi_device):
        V, policy = RVI_SVMDP(
            args=args,
            svmdp=svmdp,
            s0=s_init,
            max_iterations=10000,
            epsilon=1e-6,
            RND_SWEEPS=False,
            BATCH_SIZE=10000,
        )
    logger.info('SVMDP value iteration took %.3f sec.', time.time() - t)

    s0 = partition.x2state(model.x0)[0]
    logger.info('=== SVMDP value in initial state s0=%d: %.6f ===', s0, V[s0])

    # %% Extract policy inputs

    float_dtype = getattr(args, 'floatprecision', np.float32)
    actions_np = np.array(partition.regions['actions'])
    # RectangularPartition stores (1, num_actions, action_dim); broadcast to all states.
    if actions_np.shape[0] == 1:
        actions_np = np.broadcast_to(actions_np, (svmdp.nr_states - 1, *actions_np.shape[1:]))
    policy_inputs = np.full(
        (svmdp.nr_states - 1, actions_np.shape[2]), fill_value=float('nan'), dtype=float_dtype
    )
    mask = policy[:-1] >= 0
    policy_inputs[mask] = actions_np[mask, policy[:-1][mask]]

    # %% Simulations and plots

    from core.validate.simulate import MonteCarloSim
    from core.plotting.traces import plot_traces
    from core.plotting.heatmap import heatmap

    sim = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, verbose=False, iterations=100)
    logger.info('Empirical satisfaction probability: %s', sim.results['satprob'])

    plot_traces(
        args, stamp, model.plot_dimensions, partition, model,
        sim.results['traces'], line=False, num_traces=10, add_unsafe_box=False,
    )
    heatmap(
        args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n),
        partition=partition, results=V, filename='heatmap_satprob',
    )
    heatmap(
        args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n),
        partition=partition, results=policy_inputs[:, 0], filename='heatmap_inputs',
    )

    if args.model == 'Pendulum':
        model.plot_trajectory_gif(
            np.array(sim.results['traces'][0]['x'])[:, 0],
            filename=str(args.output_dir / f'pendulum_{stamp}.gif'),
        )

    if args.model == 'MountainCar':
        model.plot_trajectory_gif(
            np.array(sim.results['traces'][0]['x'])[:, 0],
            filename=str(args.output_dir / f'mountaincar_{stamp}.gif'),
        )
