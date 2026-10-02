import logging
from functools import partial
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import time
import argparse
from typing import Optional, Tuple

from core.abstraction.svmdp.svmdp import SVMDP

logger = logging.getLogger(__name__)

# XLA:CPU hands the minima to a YNNPACK fusion, which keeps the V gather from fusing into them: every batch
# then writes out and reads back all values of its union boxes. Without it the DP is ~3x faster.
_COMPILER_OPTIONS = {'xla_cpu_experimental_ynn_fusion_type': ''}


def _interval_minima(values, starts, lengths, axis, width):
    """
    Minima of `values` over intervals of cells along one axis.

    :param values: Array whose `axis` holds the cells of the union of the intervals (padded at the end by
        repeating its last cell)
    :param starts: First cell of every interval, relative to the union (int array, shape [m])
    :param lengths: Number of cells of every interval (int array, shape [m]), at most `width`
    :param axis: Axis to reduce (static)
    :param width: Largest interval length (static)
    :return: `values` with `axis` replaced by the minima over the m intervals
    """
    x = jnp.moveaxis(values, axis, -1)
    # Running minima: windows[j][..., u] is the minimum over the cells u, ..., u + j
    windows = [x]
    for j in range(1, width):
        shifted = jnp.concatenate([x[..., j:], jnp.repeat(x[..., -1:], j, axis=-1)], axis=-1)
        windows.append(jnp.minimum(windows[-1], shifted))
    minima = jnp.stack(windows, axis=-2)[..., lengths - 1, starts]
    return jnp.moveaxis(minima, -1, axis)


def _batch_updates(grid, slots, union_span, max_span):
    """
    Batched policy improvement and evaluation that write the batch's new values into V.

    The successor boxes of a state-action pair are all combinations of one successor interval per dimension.
    V is read once over the union box of the intervals; the minimum over every successor box then follows by
    reducing one dimension at a time to the minima over its intervals, so the boxes are never formed. The
    robust value is the boxes' probability-weighted sum of these minima.

    :param grid: TiledGrid on which V is kept
    :param slots: Number of successor intervals per dimension (tuple of D ints)
    :param union_span: Static span of the union box per dimension (tuple of D ints)
    :param max_span: Largest interval span per dimension (tuple of D ints)
    :return: Tuple (improve_batch, evaluate_batch) of jitted functions; V (their first argument) is donated
    """
    D = len(slots)
    bounds = np.cumsum((0,) + tuple(slots))
    # Dimensions with a single interval (no noise) reduce to a plain minimum; reduce them first
    order = sorted(range(D), key=lambda d: slots[d] > 1)

    def value(lb, ub, probs, V, directory):
        '''Robust value of one state-action pair (lb, ub: [sum(slots)]; probs: [prod(slots)]).'''
        lb, ub = lb.astype(jnp.int32), ub.astype(jnp.int32)
        cols, starts, lengths = [], [], []
        for d in range(D):
            lb_d, ub_d = lb[bounds[d]:bounds[d + 1]], ub[bounds[d]:bounds[d + 1]]
            # The intervals are ascending (unused slots repeat the last one), so their union is
            # [lb_d[0], ub_d[-1]]; its cells are padded to the static span by repeating the last one
            cols.append(jnp.minimum(jnp.arange(union_span[d]) + lb_d[0], ub_d[-1]))
            starts.append(lb_d - lb_d[0])
            lengths.append(ub_d - lb_d + 1)

        minima = V[grid.positions(cols, directory)]
        for d in order:
            if slots[d] == 1:
                minima = jnp.min(minima, axis=d, keepdims=True)
            else:
                minima = _interval_minima(minima, starts[d], lengths[d], d, max_span[d])

        # Boxes with an unused slot have probability 0
        return jnp.clip(probs @ minima.reshape(-1), 0.0, 1.0)

    def improvement(lb, ub, probs, V, directory):
        values = jax.vmap(value, in_axes=(0, 0, 0, None, None))(lb, ub, probs, V, directory)
        return jnp.max(values), jnp.argmax(values)

    improve = jax.vmap(improvement, in_axes=(0, 0, 0, None, None))
    evaluate = jax.vmap(value, in_axes=(0, 0, 0, None, None))

    # V is donated, so XLA writes the batch's values in place instead of copying the whole value array
    # after every batch. Callers must not use a V after passing it in.
    @partial(jax.jit, donate_argnums=(0,), compiler_options=_COMPILER_OPTIONS)
    def improve_batch(V, directory, pos, lb, ub, probs):
        values, actions = improve(lb, ub, probs, V, directory)
        return V.at[pos].set(values), actions

    @partial(jax.jit, donate_argnums=(0,), compiler_options=_COMPILER_OPTIONS)
    def evaluate_batch(V, directory, pos, lb, ub, probs):
        return V.at[pos].set(evaluate(lb, ub, probs, V, directory))

    return improve_batch, evaluate_batch


@jax.jit
def _read(V, pos):
    return V[pos]


def SVMDP_DP(
    args: argparse.Namespace,
    svmdp: SVMDP,
    s0: int,
    max_iterations: int = 1000,
    epsilon: float = 1e-6,
    sweep_priority: Optional[np.ndarray] = None,
    BATCH_SIZE: int = 8192,
    policy_iteration: bool = False,
    phase1_initial_it: int = 10,
    phase1_increment_it: int = 10,
    phase1_max_it: int = 31,
    max_eval_it: int = 31,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Robust value iteration for set-valued MDPs.

    States are updated in batches, Gauss-Seidel across batches and Jacobi within a batch. During the DP the
    values are kept on the tiled grid of the partition (svmdp.grid), where every cell outside the partition
    (the absorbing state) reads 0.

    :param args: Argument namespace
    :param svmdp: Instance of SVMDP class. Its successor arrays are released when the DP returns.
    :param s0: Initial state for tracking
    :param max_iterations: Maximum number of iterations
    :param epsilon: Convergence threshold
    :param sweep_priority: Optional per-state priority (e.g. steps-to-goal along the RL rollouts); states are
        swept in ascending priority, with ties in random order. Without it, the order is random.
    :param BATCH_SIZE: Batch size for state updates
    :param policy_iteration: Whether to use policy iteration instead of value iteration
    :param phase1_initial_it: Base cap on inner policy-evaluation sweeps in the first outer iteration
    :param phase1_increment_it: Per-outer-iteration growth of the inner-sweep cap
    :param phase1_max_it: Hard ceiling on the inner-sweep cap (before full convergence)
    :param max_eval_it: Maximum evaluation iteration index (keeps eval_it <= max_eval_it, strictly < 32 by default)
    :return: Tuple of (values, policy_labels) where policy_labels[s] is the global action ID chosen for state s, or -1
    """

    start_time = time.time()
    grid = svmdp.grid

    # Count the total number of actions. A_id is a single shared list of action ids:
    # every state has the same actions enabled.
    num_actions = len(svmdp.A_id)

    if policy_iteration:
        logger.info('(Algorithm: robust policy iteration)')
    else:
        logger.info('(Algorithm: robust value iteration)')

    logger.info('- Number of states: %d', len(svmdp.states))
    logger.info('- Total number of choices: %d (total number of state-action pairs)', len(svmdp.states) * num_actions)
    logger.info('- Max number of actions per state: %d', num_actions)
    logger.info('- Batch size for dynamic programming over states: %d', BATCH_SIZE)
    logger.info(f'- Successor intervals per dimension: {svmdp.slots} -> {int(np.prod(svmdp.slots))} boxes per '
                f'state-action, read over a union box of {svmdp.union_span} = {int(np.prod(svmdp.union_span))} cells')

    #####

    logger.info('- Set states to update...')
    # Goal and critical (unsafe) states keep their values (1 and 0); all other states are updated
    skip_mask = svmdp.critical_regions | svmdp.goal_regions
    states_to_update = svmdp.states[~skip_mask]
    states_not_to_update = svmdp.states[skip_mask]
    logger.info(f'  (Active states in initial mask: {len(states_to_update)})')

    # Initialize value function (on the tiled grid) and policy
    V_grid = np.zeros(grid.size, dtype=args.floatprecision)
    V_grid[grid.state_pos[svmdp.goal_regions]] = 1.0
    V = jnp.array(V_grid)
    del V_grid

    policy = np.zeros(svmdp.nr_states, dtype=np.int32)
    policy[states_not_to_update] = -1  # Mark states that we do not update with a special action index (e.g., -1)

    # Sweep order: random, or by sweep priority (states nearest the goal first, so one Gauss-Seidel sweep
    # carries value back along the RL rollouts); the stable sort keeps the random order among equal priorities.
    perm = np.random.permutation(len(states_to_update))
    if sweep_priority is not None:
        perm = perm[np.argsort(sweep_priority[states_to_update[perm]], kind='stable')]
    logger.info('- State sweep order: %s', 'by sweep priority' if sweep_priority is not None else 'random')
    states_to_update = states_to_update[perm]

    # Batches in sweep order. Within a batch the order does not matter (Jacobi), so its states are sorted,
    # which speeds up reading their rows of the successor arrays. The rows are read from svmdp at every
    # improvement sweep, so the multi-GB successor arrays are held only once.
    state_batches = [np.sort(states_to_update[i:i + BATCH_SIZE]) for i in range(0, len(states_to_update), BATCH_SIZE)]
    pos_batches = [jnp.asarray(grid.state_pos[sb]) for sb in state_batches]
    pos_update = jnp.asarray(grid.state_pos[states_to_update])
    pos_s0 = int(grid.state_pos[s0])
    directory = grid.directory

    improve_batch, evaluate_batch = _batch_updates(grid, svmdp.slots, svmdp.union_span, svmdp.max_slice)

    def improvement_sweep(V):
        '''Policy improvement over all states (in sweep order); updates the policy, returns the new V.'''
        actions = []
        for sb, pos in zip(state_batches, pos_batches):
            V, a = improve_batch(V, directory, pos, svmdp.interval_lb[sb], svmdp.interval_ub[sb], svmdp.box_probs[sb])
            actions.append(a)
        for sb, a in zip(state_batches, jax.device_get(actions)):
            policy[sb] = np.asarray(a, dtype=np.int32)
        return V

    logger.info(f'- SVMDP defined (took {time.time() - start_time:.3f}s)')
    start_time = time.time()

    satprob = args.satprob
    Vs = _read(V, pos_update)  # Values of the updated states, for the convergence checks
    pbar = tqdm(desc='Iteration', total=None, unit='it', dynamic_ncols=True, leave=True)
    if not policy_iteration:
        # Value iteration
        for iteration in range(max_iterations):
            pbar.update(1)
            pbar.set_postfix({f'v[{s0}]': f'{float(V[pos_s0]):.8f}', 'v_avg': f'{float(jnp.mean(Vs)):.8f}'})

            # Policy evaluation + improvement (later batches see earlier updates)
            Vs_old = Vs
            V = improvement_sweep(V)
            Vs = _read(V, pos_update)

            if float(V[pos_s0]) >= satprob:
                pbar.write(f'Threshold reached: v[{s0}]={float(V[pos_s0]):.8f} >= {satprob} after {iteration + 1} iterations')
                break

            # Check convergence
            if float(jnp.max(jnp.abs(Vs - Vs_old))) < epsilon:
                pbar.write(f'Converged after {iteration + 1} iterations')
                break

    else:
        partial_convergence_reached = False
        sat_policy = False
        delta = float('inf')

        # eval_batches holds the policy-selected successor inputs per batch; eval_policy_state[s] records which
        # action is currently reflected for state s, so only the rows whose policy changed are re-gathered.
        eval_batches = [None] * len(state_batches)
        eval_policy_state = np.empty(svmdp.nr_states, dtype=np.int32)

        # Policy iteration
        for iteration in range(max_iterations):

            pbar.update(1)

            # Policy evaluation. The policy is fixed throughout, so gather the policy-selected action's inputs
            # once here, rather than on every inner sweep.
            i = 0
            for bi, sb in enumerate(state_batches):
                sel = policy[sb]
                if eval_batches[bi] is None:
                    eval_batches[bi] = [svmdp.interval_lb[sb, sel], svmdp.interval_ub[sb, sel], svmdp.box_probs[sb, sel]]
                else:
                    changed = np.flatnonzero(sel != eval_policy_state[sb])
                    if changed.size:
                        rows, acts = sb[changed], sel[changed]
                        ev_lb, ev_ub, ev_p = eval_batches[bi]
                        ev_lb[changed] = svmdp.interval_lb[rows, acts]
                        ev_ub[changed] = svmdp.interval_ub[rows, acts]
                        ev_p[changed] = svmdp.box_probs[rows, acts]
                eval_policy_state[sb] = sel

            while True:

                v_s0 = float(V[pos_s0])
                postfix_dict = {
                    'eval_it': f'{i}',
                    f'v[{s0}]': f'{v_s0:.8f}',
                    'v_avg': f'{float(jnp.mean(Vs)):.8f}',
                    'max(v-v_old)': f'{delta:.8f}',
                }

                # Check if policy is above the preset threshold quality
                if v_s0 >= satprob:
                    logger.info(f'Policy is above the satisfaction threshold {satprob:.2f} after {iteration + 1} iterations')
                    # Policy is already good enough, so skip policy improvement and only keep evaluating it until convergence
                    sat_policy = True
                else:
                    sat_policy = False
                pbar.set_postfix(postfix_dict)

                # Policy evaluation only (later batches see earlier updates)
                Vs_old = Vs
                for pos, (ev_lb, ev_ub, ev_p) in zip(pos_batches, eval_batches):
                    V = evaluate_batch(V, directory, pos, ev_lb, ev_ub, ev_p)
                Vs = _read(V, pos_update)

                delta = float(jnp.max(jnp.abs(Vs - Vs_old)))
                if (
                    delta < epsilon
                    or i >= max_eval_it
                    or (float(V[pos_s0]) >= satprob)
                    or (
                        not partial_convergence_reached
                        and i >= min(phase1_initial_it + iteration * phase1_increment_it, phase1_max_it)
                    )
                ):
                    break

                i += 1

            # Policy evaluation + improvement
            Vs_before_improvement = Vs
            if not sat_policy:
                V = improvement_sweep(V)
                Vs = _read(V, pos_update)

            if float(V[pos_s0]) >= satprob:
                pbar.write(f'Threshold reached: v[{s0}]={float(V[pos_s0]):.8f} >= {satprob} after {iteration + 1} iterations')
                break

            # Check convergence: improvement step is monotone, so max gain suffices
            # TODO: Better validate the convergence criterion based on max gain (rather than checking if the policy is unchanged; which is less stable in case of multiple optimal policies)
            if float(jnp.max(Vs - Vs_before_improvement)) < epsilon:
                if partial_convergence_reached:
                    pbar.write(f'Converged after {iteration + 1} iterations')
                    break
                else:
                    pbar.write(
                        f'Partial convergence after {iteration + 1} iterations. '
                        'Decrease epsilon to refine values...'
                    )
                    partial_convergence_reached = True

    pbar.close()

    # Values of the partition states, and of the absorbing state (0)
    values = np.zeros(svmdp.nr_states, dtype=args.floatprecision)
    values[:-1] = np.asarray(_read(V, jnp.asarray(grid.state_pos)))

    # The successor arrays are not used after the DP; release them (they are also left out of checkpoints)
    svmdp.interval_lb = svmdp.interval_ub = svmdp.box_probs = None

    # Extract policy inputs from policy. A_id is the shared list of action ids, so the
    # chosen action index maps directly through it (identity when A_id == range(num_actions)).
    A_id_arr = np.asarray(svmdp.A_id)
    policy_labels = np.full_like(policy, fill_value=-1)
    valid = policy != -1
    policy_labels[valid] = A_id_arr[policy[valid]]

    logger.info(f'- Policy synthesis finished (took {time.time() - start_time:.3f}s)\n')

    return values, policy_labels
