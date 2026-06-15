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

    args.model = 'Drone4D'
    args.decimals = 9
    args.pAbs_min = 0

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

        active_states, active_actions = find_active(model, args=args, previous_cells=set())
        logger.info(f"Identified {len(active_states)} active states from RL exploration.\n")

        # Create partition of the continuous state space into convex polytope
        # partition = RectangularPartition(model=model)
        # Sparse partition can be created with, e.g.,
        partition = SparsePartition(model=model, active_states=active_states, active_actions=active_actions)

        s_init_debug, s_init_exists = partition.x2state(model.x0)

        # Create actions based on forward reachable sets
        actions = RectangularForward(args=args, partition=partition, model=model)
        
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
             
        s = s_init_debug

        frs_lb = actions.frs_lb[s][0]
        print('frs_lb: ', frs_lb)
        frs_ub = actions.frs_ub[s][0]
        print('frs_ub: ', frs_ub)

        frs_idx_lb = actions.frs_idx_lb[s][0]
        print('frs_idx_lb: ', frs_idx_lb)

        frs_idx_ub = frs_idx_lb + actions.max_slice - 1
        print('frs_idx_ub: ', frs_idx_ub)

        import math
        from scipy.stats import norm

        # x dimension
        mu1, sigma = frs_lb[0], 0.15  # Standard normal distribution
        mu2, sigma = frs_ub[0], 0.15  # Standard normal distribution
        a, b = partition.regions_per_dim['lower_bounds'][0][frs_idx_lb[0]], partition.regions_per_dim['upper_bounds'][0][frs_idx_ub[0]]
        prob_x1 = norm.cdf(b, mu1, sigma) - norm.cdf(a, mu1, sigma)
        prob_x2 = norm.cdf(b, mu2, sigma) - norm.cdf(a, mu2, sigma)
        print('Min probability for x dimension: ', min(prob_x1, prob_x2))
        print('Max probability for x dimension: ', max(prob_x1, prob_x2))

        # y dimension
        mu1, sigma = frs_lb[2], 0.15  # Standard normal distribution
        mu2, sigma = frs_ub[2], 0.15  # Standard normal distribution
        a, b = partition.regions_per_dim['lower_bounds'][2][frs_idx_lb[2]], partition.regions_per_dim['upper_bounds'][2][frs_idx_ub[2]]
        prob_y1 = norm.cdf(b, mu1, sigma) - norm.cdf(a, mu1, sigma)
        prob_y2 = norm.cdf(b, mu2, sigma) - norm.cdf(a, mu2, sigma)
        print('Min probability for y dimension: ', min(prob_y1, prob_y2))
        print('Max probability for y dimension: ', max(prob_y1, prob_y2))

        print('Min prob to reach absorbing: ', 1 - max(prob_x1, prob_x2) * max(prob_y1, prob_y2))
        print('Max prob to reach absorbing: ', 1 - min(prob_x1, prob_x2) * min(prob_y1, prob_y2))
        
        print('Computed P_absorbing: ', P_absorbing[s][0])
        print()

        # --- Transition probability intervals for initial state (debug) ---
        # s0 = s_init_debug
        # enabled_action_ids = A_id.get(s0)
        # dbg_critical = np.array(partition.critical['bools'])
        # dbg_goal     = np.array(partition.goal['bools'])
        # absorbing_state  = int(np.max(partition.regions['idxs'])) + 1
        # actions_arr = np.asarray(partition.regions['actions'])
        # # Sparse partition: actions_arr[s, a]; rectangular: actions_arr[0, a]
        # state_row = s0 if actions_arr.shape[0] > 1 else 0
        # print(f"\n=== Transition intervals for s0={s0} ===")
        # if enabled_action_ids is None or len(enabled_action_ids) == 0:
        #     print(f"  No enabled actions for s0={s0}.")
        # else:
        #     for local_idx, global_aid in enumerate(enabled_action_ids):
        #         u = actions_arr[state_row, global_aid]
        #         probs    = P_full[s0][local_idx]      # shape [num_successors, 2]
        #         succ_ids = S_id[s0][local_idx]         # shape [num_successors]
        #         p_abs    = P_absorbing[s0][local_idx]  # shape [2]
        #         print(f"Action {global_aid} (u={np.array(u)}):")
        #         print(f"  state {absorbing_state}: P=[{p_abs[0]:.6f}, {p_abs[1]:.6f}]  [ABSORBING]")
        #         for k in range(len(probs)):
        #             if probs[k, 1] == 0:
        #                 continue
        #             sid = int(succ_ids[k])
        #             is_critical = bool(dbg_critical[sid]) if sid < len(dbg_critical) else False
        #             is_goal     = bool(dbg_goal[sid])     if sid < len(dbg_goal)     else False
        #             is_selfloop = (sid == s0)
        #             tag = "  [ABSORBING - critical]" if is_critical else ("  [ABSORBING - goal]" if is_goal else ("  [self-loop]" if is_selfloop else ""))
        #             print(f"  state {sid}: P=[{probs[k, 0]:.6f}, {probs[k, 1]:.6f}]{tag}")
        # print("=== End ===")

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
    if args.solver == 'jax':
        t = time.time()
        with jax.default_device(args.rvi_device):
            V, policy = RVI_JAX(
                args=args,
                imdp=imdp,
                s0=partition.x2state(model.x0)[0],
                max_iterations=10000,
                epsilon=1e-6,
                RND_SWEEPS=True,
                BATCH_SIZE=10000,
                policy_iteration=args.policy_iteration,
            )
        logger.info('RVI with JAX (random-batched asynchronous) took %.3f sec.', (time.time() - t))
    else:
        t = time.time()
        V, policy = RVI_STORM(
            args=args,
            imdp=imdp,
        )
        logger.info('RVI with Storm took %.3f sec.', (time.time() - t))

    # Extract policy
    float_dtype = getattr(args, "floatprecision", np.float32)
    # Define concrete policy (but exclude final IMDP state, which is absorbing and has no actions)
    actions_np = np.array(partition.regions['actions'])
    # RectangularPartition stores (1, num_actions, action_dim); SparsePartition stores (num_states, ...).
    if actions_np.shape[0] == 1:
        actions_np = np.broadcast_to(actions_np, (imdp.nr_states - 1, *actions_np.shape[1:]))
    policy_inputs = np.full((imdp.nr_states - 1, actions_np.shape[2]), fill_value=float('nan'), dtype=float_dtype)
    mask = policy[:-1] >= 0
    policy_inputs[mask] = actions_np[mask, policy[:-1][mask]]

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