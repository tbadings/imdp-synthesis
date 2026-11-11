import copy
import csv
import datetime
import logging
import os
import pickle
import random
import time
from pathlib import Path
import jax
import numpy as np

import benchmarks
from core.abstraction.svmdp.forward_reachability import RectangularForward
from core.abstraction.svmdp.successor_ids import make_box_to_ids
from core.abstraction.svmdp.svmdp import SVMDP
from core.abstraction.svmdp.dynprog import SVMDP_DP
from core.options import parse_arguments
from core.abstraction.partition import DensePartition, SparsePartition
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
    args.jax_key = jax.random.PRNGKey(args.seed)

    # Set current working directory
    args.cwd = os.path.dirname(os.path.abspath(__file__))
    args.root_dir = Path(args.cwd)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dict = {}
    planning_reference = None

    ckpt = {}
    if args.load_checkpoint:
        # --- Load SVMDP from checkpoint ---
        ckpt_path = Path(args.load_checkpoint)
        logger.info('Loading checkpoint from %s', ckpt_path)
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        model = ckpt['model']
        partition = ckpt['partition']
        svmdp = ckpt['svmdp']
        args.model = ckpt['args'].model
        args.model_version = getattr(ckpt['args'], 'model_version', args.model_version)
        out_dict.update(ckpt.get('out_dict', {}))
        planning_reference = ckpt.get('planning_reference')

        run_output_dir = args.root_dir / args.output_root / f"{stamp}_{args.model}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = run_output_dir
        add_file_handler(run_output_dir, stamp)
        logger.info('Run %s | model=%s (from checkpoint)', stamp, args.model)
        logger.info('Output directory: %s', run_output_dir)

        logger.info('\n=== SVMDP loaded from checkpoint: %s ===', ckpt_path)
    else:
        t = time.time()
        
        # --- Build SVMDP from scratch ---
        run_output_dir = args.root_dir / args.output_root / f"{stamp}_{args.model}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = run_output_dir
        add_file_handler(run_output_dir, stamp)
        logger.info('Run %s | model=%s | noise=%s', stamp, args.model, args.noise_distr)
        logger.info('Output directory: %s', run_output_dir)

        logger.info('\n=== Generate SVMDP from scratch ===')
        logger.debug('Arguments: %s', vars(args))

        model = benchmarks.create_model(args)

        if args.dense:
            logger.info('Using DensePartition (RL exploration skipped).')
            partition = DensePartition(model=model)
        elif args.rrt:
            from core.rrt import find_active as find_planned_states

            active_states, active_actions, planning_reference, metrics = find_planned_states(model, args)
            out_dict.update(metrics)
            t = time.time()
            partition = SparsePartition(model=model, active_states=active_states, active_actions=active_actions)
            out_dict['exploration'] = 'rrt'
            del active_states, active_actions
        else:
            active_states, active_actions, _ = find_active(model, args=args)
            logger.info(f"Identified {len(active_states)} active states from RL exploration.\n")

            out_dict['time_RL'] = time.time() - t
            logger.info('<<< Generating model and running RL took %.3f sec. >>>\n', out_dict['time_RL'])
            t = time.time()

            # Create partition of the continuous state space into convex polytope
            partition = SparsePartition(model=model, active_states=active_states, active_actions=active_actions)

            # The RL exploration outputs are only needed to build the partition; free them
            # (and let the PPO model / vec envs held internally be reclaimed) before the
            # large forward-reachability arrays are allocated below.
            del active_states, active_actions

        x0_box = getattr(model, 'x0_box', None)
        S0, S0_exist = partition.initial_states(model.x0, x0_box)
        if not S0_exist:
            raise ValueError(f"Initial state x0={model.x0} (x0_box={x0_box}) is not (fully) in the partition.")

        # Compute forward reachable sets and noise-shifted successor cell IDs.
        actions = RectangularForward(args=args, partition=partition, model=model)

        # All partition states have all actions enabled (rectangular partition),
        # so a single shared list of action ids describes every state.
        states = np.array(partition.regions['idxs'])
        A_id = list(range(actions.num_actions))

        # Recompose successor IDs on the fly in the DP from the compact boxes (frs_idx_lb/frs_idx_ub)
        # rather than materialising the [S, A, nc, prod(max_span)] ID array (tens of GB for 3-D models).
        box_to_ids = make_box_to_ids(max_span=actions.max_slice, wrap=model.wrap, partition=partition)

        svmdp = SVMDP(
            partition=partition,
            states=states,
            x0=model.x0,
            goal_regions=np.array(partition.goal['bools']),
            critical_regions=np.array(partition.critical['bools']),
            P_full=actions.frs_noise_probs,
            S_idx_lb=actions.frs_idx_lb,
            S_idx_ub=actions.frs_idx_ub,
            box_to_ids=box_to_ids,  
            A_id=A_id,
            P_absorbing=model.noise.partition['remainder'],
            x0_box=x0_box,
        )

        del actions

        out_dict['time_abstraction'] = time.time() - t
        logger.info('Initial state x0=%s (x0_box=%s) → %d state(s): %s\n', model.x0, x0_box, len(S0), S0[:10])
        logger.info('<<< Generating SVMDP abstraction took %.3f sec. >>>\n', out_dict['time_abstraction'])

        out_dict['abstraction_states'] = len(svmdp.states)
        out_dict['abstraction_actions'] = len(svmdp.A_id)
        out_dict['abstraction_state-actions'] = len(svmdp.states) * len(svmdp.A_id)

    # %% Run value iteration on the SVMDP

    if args.load_checkpoint:
        V = ckpt['V']
        policy = ckpt['policy']
        logger.info('Loaded synthesized values and policy; skipping policy synthesis.')
    else:
        logger.info('=== SVMDP policy synthesis ===')
        t = time.time()
        with jax.default_device(args.rvi_device):
            V, policy = SVMDP_DP(
                args=args,
                svmdp=svmdp,
                s0=svmdp.s_init,
                max_iterations=10000,
                epsilon=1e-6,
                RND_SWEEPS=True,
                BATCH_SIZE=1000,
                policy_iteration=args.policy_iteration,
                prune_states=False
            )

        out_dict['time_synthesis'] = time.time() - t
        logger.info('<<< SVMDP policy synthesis done (took %.3f sec.) >>>\n', out_dict['time_synthesis'])

    # Guarantee for the initial state(s) = minimum value over all initial cells
    S0 = np.atleast_1d(svmdp.s_init)
    out_dict['optimal_value'] = float(V[S0].min())
    logger.info('Value in initial state (min over %d cells): %.6f\n', len(S0), V[S0].min())

    # %% Extract policy inputs

    if not args.load_checkpoint:
        float_dtype = getattr(args, 'floatprecision', np.float32)
        actions_np = np.array(partition.regions['actions'])
        # DensePartition stores (1, num_actions, action_dim); broadcast to all states.
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
    from core.plotting.traces import plot_traces_3d

    if args.load_checkpoint:
        sim_results = ckpt['sim_results']
        logger.info('Loaded simulation traces; skipping Monte Carlo simulations.')
    elif not args.mc_simulations:
        sim_results = {'satprob': None, 'traces': {}}
        logger.info('Skipping Monte Carlo simulations (--no-mc_simulations).')
    else:
        sim = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, verbose=False, iterations=1000)
        sim_results = sim.results
        del sim

    if sim_results['satprob'] is not None:
        out_dict['empirical_satprob'] = sim_results['satprob']
        logger.info('Empirical satisfaction probability: %s', sim_results['satprob'])

    # Save completed synthesis and validation before plotting, so plots can be regenerated
    # directly (and a plotting failure does not lose the expensive computation).
    if args.save_checkpoint:
        args_to_save = copy.copy(args)
        del args_to_save.rvi_device
        del args_to_save.jax_key
        ckpt_path = args.output_dir / 'checkpoint.pkl'
        logger.info('Saving completed checkpoint to %s', ckpt_path)
        with open(ckpt_path, 'wb') as f:
            pickle.dump({
                'model': model, 'partition': partition, 'svmdp': svmdp,
                'args': args_to_save, 'V': np.asarray(V), 'policy': np.asarray(policy),
                'sim_results': sim_results, 'out_dict': out_dict,
                'planning_reference': planning_reference,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info('Checkpoint saved.\n')

    # Export output dictionary to csv (all run arguments and the benchmark's initial set first, then the metrics)
    x0_box = getattr(model, 'x0_box', None)
    summary = {
        **vars(args),
        'x0': np.asarray(model.x0).tolist(),
        'x0_box': None if x0_box is None else np.asarray(x0_box).tolist(),
        **out_dict,
    }
    csv_path = args.output_dir / 'summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['metric', 'value'])
        writer.writerows(summary.items())
    logger.info('Run summary saved to %s', csv_path)

    heatmap(
        args, stamp, idx_show=model.plot_dimensions,
        partition=partition, results=V, filename='heatmap_satprob',
        model=model,
    )
    if sim_results['traces']:
        plot_traces(
            args, stamp, model.plot_dimensions, partition, model,
            sim_results['traces'], line=False, num_traces=100, add_unsafe_box=False,
        )

        if args.model.startswith('Drone6D'):
            print('Plot Drone6D traces in 3D...')
            plot_traces_3d(
                args, stamp, [0, 2, 4], partition, model,
                sim_results['traces'], num_traces=100, filename="traces_3d",
            )
            from core.plotting.drone3d import plot_drone_3d_pyvista
            plot_drone_3d_pyvista(args, stamp, [0, 2, 4], partition, model,
                                  sim_results['traces'], num_traces=10)

        if args.model == 'Pendulum':
            print('Plot Pendulum gif...')
            model.plot_trajectory_gif(
                np.array(sim_results['traces'][0]['x'])[:, 0],
                filename=str(args.output_dir / f'pendulum_{stamp}.gif'),
            )

        if args.model == 'MountainCar':
            print('Plot MountainCar gif...')
            model.plot_trajectory_gif(
                np.array(sim_results['traces'][0]['x'])[:, 0],
                filename=str(args.output_dir / f'mountaincar_{stamp}.gif'),
            )

        if args.model == 'CartPole':
            print('Plot CartPole gif...')
            model.plot_trajectory_gif(
                np.array(sim_results['traces'][0]['x'])[:, [0, 2]],
                filename=str(args.output_dir / f'cartpole_{stamp}.gif'),
            )
