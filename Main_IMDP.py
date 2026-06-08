import datetime
import copy
import logging
import os
import pickle
import time
from pathlib import Path
import jax
import numpy as np

import benchmarks
from core.abstraction.imdp.probability_intervals import compute_probability_intervals
from core.abstraction.imdp.forward_reachability import RectangularForward
from core.options import parse_arguments
from core.abstraction.partition import RectangularPartition, SparsePartition
from core.abstraction.imdp.imdp import IMDP
from core.abstraction.imdp.rvi_jax import RVI_JAX
from core.abstraction.imdp.rvi_storm import RVI_STORM
from core.jax_config import configure_jax
from core.utils import configure_logging, add_file_handler
from core.rl import find_active

if __name__ == '__main__':
    args = parse_arguments()
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    configure_jax(args)

    np.random.seed(args.seed)
    args.jax_key = jax.random.PRNGKey(args.seed)

    # Set current working directory
    args.cwd = os.path.dirname(os.path.abspath(__file__))
    args.root_dir = Path(args.cwd)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.load_checkpoint:
        # --- Load IMDP from checkpoint ---
        ckpt_path = Path(args.load_checkpoint)
        logger.info('Loading checkpoint from %s', ckpt_path)
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        model = ckpt['model']
        partition = ckpt['partition']
        imdp = ckpt['imdp']
        args.model = ckpt['args'].model

        run_output_dir = args.root_dir / args.output_root / f"{stamp}_{args.model}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = run_output_dir
        add_file_handler(run_output_dir, stamp)
        logger.info('Run %s | model=%s (from checkpoint)', stamp, args.model)
        logger.info('Output directory: %s', run_output_dir)

        logger.info('\n=== IMDP loaded from checkpoint: %s ===', ckpt_path)
    else:
        # --- Build IMDP from scratch ---
        run_output_dir = args.root_dir / args.output_root / f"{stamp}_{args.model}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = run_output_dir
        add_file_handler(run_output_dir, stamp)
        logger.info('Run %s | model=%s | noise=%s | batch=%d', stamp, args.model, args.noise_distr, args.batch_size)
        logger.info('Output directory: %s', run_output_dir)

        logger.info('\n=== Generating IMDP from scratch ===')
        logger.debug('Arguments: %s', vars(args))

        # Define and parse model
        model = benchmarks.create_model(args)

        t = time.time()

        active_states = find_active(model, args=args, previous_cells=set())
        print(f"Identified {len(active_states)} active states from RL exploration.")
        # Create partition of the continuous state space into convex polytope
        # partition = RectangularPartition(model=model)
        # Sparse partition can be created with, e.g.,
        partition = SparsePartition(model=model, active_states=active_states)

        s_init_debug, s_init_exists = partition.x2state(model.x0)

        # Create actions based on forward reachable sets
        actions = RectangularForward(args=args, partition=partition, model=model)
        actions_inputs = actions.id_to_input

        if not s_init_exists:
            raise ValueError(f"Initial state x0={model.x0} is not an active cell in the partition.")
        # print(f"\n=== Forward reachable sets for initial state s0={s_init_debug} (x0={model.x0}) ===")
        # for a_idx in range(len(actions.id_to_input)):
        #     u = actions.id_to_input[a_idx]
        #     lb = actions.frs_lb[s_init_debug, a_idx]
        #     ub = actions.frs_ub[s_init_debug, a_idx]
        #     idx_lb = actions.frs_idx_lb[s_init_debug, a_idx]
        #     print(f"  action {a_idx:3d}: u={np.array(u)}  FRS=[{np.array(lb)}, {np.array(ub)}]  grid_idx_lb={np.array(idx_lb)}")
        # print("=== End forward reachable sets ===\n")

        logger.info('Initial state x0=%s maps to state index %d (exists in partition: %s)',
                    model.x0, s_init_debug, s_init_exists)
        P_full, S_id, A_id, P_absorbing = compute_probability_intervals(args=args,
                                                        model=model,
                                                        partition=partition,
                                                        actions=actions,
                                                        vectorized=True,
                                                        debug_state=s_init_debug if s_init_exists else None)

        # --- Transition probability intervals for neutral action u=0 in initial state ---
        s0 = s_init_debug
        enabled_action_ids = A_id.get(s0, np.array([]))
        critical_regions = np.array(partition.critical['bools'])
        goal_regions     = np.array(partition.goal['bools'])
        absorbing_state  = int(np.max(partition.regions['idxs'])) + 1
        print(f"\n=== Transition intervals for s0={s0}, neutral action u=0 ===")
        found = False
        for local_idx, global_aid in enumerate(enabled_action_ids):
            u = actions.id_to_input[global_aid]
            if not np.allclose(u, 0):
                continue
            found = True
            probs    = P_full[s0][local_idx]      # shape [num_successors, 2]
            succ_ids = S_id[s0][local_idx]         # shape [num_successors]
            p_abs    = P_absorbing[s0][local_idx]  # shape [2]
            print(f"Action {global_aid} (u={np.array(u)}):")
            print(f"  state {absorbing_state}: P=[{p_abs[0]:.6f}, {p_abs[1]:.6f}]  [ABSORBING]")
            for k in range(len(probs)):
                if probs[k, 1] == 0:
                    continue
                sid = int(succ_ids[k])
                is_critical = bool(critical_regions[sid]) if sid < len(critical_regions) else False
                is_goal     = bool(goal_regions[sid])     if sid < len(goal_regions)     else False
                tag = ""
                if is_critical:
                    tag = "  [ABSORBING - critical]"
                elif is_goal:
                    tag = "  [ABSORBING - goal]"
                print(f"  state {sid}: P=[{probs[k,0]:.6f}, {probs[k,1]:.6f}]{tag}")
        if not found:
            print("  No neutral action (u=0) found among enabled actions for s0.")
        print("=== End ===\n")

        # assert False
        # del actions        

        # cell = (7, 16, 7, 17, 7, 16)
        # state_id, exists = partition.grid_idx2state(cell)
        # print('cell', cell)
        # print('state_id', state_id)
        # print('exists', exists)
        # labels = A_id.get(state_id)
        # print('action_labels', labels)
        # if labels is not None:
        #     labels = np.asarray(labels, dtype=int)
        #     print('num_actions', len(labels))
        #     print('action_inputs')
        #     print(actions.id_to_input[labels])
        # print(P_full[14649][1], S_id[14649][1], A_id[14649][1], P_absorbing[14649][1])



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

        if args.save_checkpoint:
            # Save checkpoint (strip JAX runtime objects that can't be pickled)
            args_to_save = copy.copy(args)
            del args_to_save.rvi_device
            del args_to_save.jax_key
            ckpt_path = args.output_dir / 'checkpoint.pkl'
            logger.info('Saving checkpoint to %s', ckpt_path)
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'model': model, 'partition': partition, 'imdp': imdp, 'args': args_to_save}, f)
            logger.info('Checkpoint saved.')

    # %% Run dynamic programming to compute optimal policy

    logger.info('\n=== Computing optimal policy via robust dynamic programming (solver=%s) ===', args.solver)
    t = time.time()
    if args.solver == 'jax':
        with jax.default_device(args.rvi_device):
            V, policy, policy_inputs = RVI_JAX(
                args=args,
                imdp=imdp,
                s0=partition.x2state(model.x0)[0],
                max_iterations=10000,
                epsilon=1e-6,
                RND_SWEEPS=True,
                BATCH_SIZE=1000,
                policy_iteration=args.policy_iteration,
            )
        logger.info('RVI with JAX (random-batched asynchronous) took %.3f sec.', (time.time() - t))
    else:
        V, policy, policy_inputs = RVI_STORM(
            args=args,
            imdp=imdp,
        )
        logger.info('RVI with Storm took %.3f sec.', (time.time() - t))

    s0 = partition.x2state(model.x0)[0]
    logger.info('=== IMDP value in initial state s0=%s: %s ===', s0, V[s0])    

    # %% Simulations and plot

    sim_policy = policy
    sim_policy_inputs = policy_inputs
    sim_values = V

    from core.validate.simulate import MonteCarloSim
    from core.plotting.traces import plot_traces
    from core.plotting.heatmap import heatmap

    sim = MonteCarloSim(model, partition, sim_policy, sim_policy_inputs, model.x0, verbose=False, iterations=100)
    logger.info('Empirical satisfaction probability: %s', sim.results['satprob'])

    plot_traces(args, stamp, model.plot_dimensions, partition, model, sim.results['traces'], line=False, num_traces=10, add_unsafe_box=False,)
    heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=sim_values, filename="heatmap_satprob")
    heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=sim_policy_inputs[:,0], filename="heatmap_inputs")
    
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