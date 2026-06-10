import argparse
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from jaxtyping import Array, Bool, Float32, UInt8

from core.abstraction.imdp.imdp import IMDP

logger = logging.getLogger(__name__)

try:
    import stormpy
except ImportError:  # pragma: no cover
    stormpy = None


def _normalize_region_mask(region: np.ndarray, nr_non_absorbing_states: int) -> np.ndarray:
    """Return a boolean mask over the non-absorbing states."""
    region_arr = np.asarray(region)
    if region_arr.dtype == bool:
        if region_arr.shape[0] != nr_non_absorbing_states:
            raise ValueError(
                "Expected boolean region mask with one entry per non-absorbing state."
            )
        return region_arr

    mask = np.zeros(nr_non_absorbing_states, dtype=bool)
    if region_arr.size > 0:
        mask[region_arr.astype(np.int32)] = True
    return mask


def _to_interval(
    interval_cache: Dict[Tuple[float, float], Any],
    lower: float,
    upper: float,
):
    key = (float(lower), float(upper))
    if key not in interval_cache:
        interval_cache[key] = stormpy.pycarl.Interval(key[0], key[1])
    return interval_cache[key]


def _build_storm_imdp(imdp: IMDP):
    if stormpy is None:
        raise ImportError(
            "stormpy is required for RVI_STORM but is not installed. "
            "Install Storm/Stormpy and retry."
        )

    nr_non_absorbing_states = len(imdp.states)
    goal_mask = _normalize_region_mask(imdp.goal_regions, nr_non_absorbing_states)
    critical_mask = _normalize_region_mask(imdp.critical_regions, nr_non_absorbing_states)

    builder = stormpy.IntervalSparseMatrixBuilder(
        rows=0,
        columns=0,
        entries=0,
        force_dimensions=False,
        has_custom_row_grouping=True,
        row_groups=0,
    )

    interval_cache: Dict[Tuple[float, float], Any] = {
        (1.0, 1.0): stormpy.pycarl.Interval(1.0, 1.0)
    }

    total_choices = 1  # absorbing state's single self-loop choice
    for s in imdp.states:
        s_int = int(s)
        has_actions = s_int in imdp.A_id and len(imdp.A_id[s_int]) > 0
        is_goal = bool(goal_mask[s_int])
        is_critical = bool(critical_mask[s_int])

        if is_goal or is_critical or not has_actions:
            total_choices += 1
        else:
            total_choices += int(len(imdp.A_id[s_int]))

    choice_labeling = stormpy.storage.ChoiceLabeling(total_choices)

    action_labels = {-1}
    for s in imdp.states:
        if s in imdp.A_id:
            action_labels.update(int(a) for a in np.asarray(imdp.A_id[s], dtype=np.int32))

    for action_label in sorted(action_labels):
        choice_labeling.add_label(str(action_label))

    row = 0
    for s in imdp.states:
        s_int = int(s)
        builder.new_row_group(row)

        has_actions = s_int in imdp.A_id and len(imdp.A_id[s_int]) > 0
        is_goal = bool(goal_mask[s_int])
        is_critical = bool(critical_mask[s_int])

        if is_critical or not has_actions:
            choice_labeling.add_label_to_choice(str(-1), row)
            builder.add_next_value(
                row,
                int(imdp.absorbing_state),
                interval_cache[(1.0, 1.0)],
            )
            row += 1
            continue

        if is_goal:
            choice_labeling.add_label_to_choice(str(-1), row)
            builder.add_next_value(row, s_int, interval_cache[(1.0, 1.0)])
            row += 1
            continue

        successors = imdp.S_id[s_int]
        for a_idx, a_label in enumerate(imdp.A_id[s_int]):
            choice_labeling.add_label_to_choice(str(int(a_label)), row)

            for s_next, prob_interval in zip(successors[a_idx], imdp.P_full[s_int][a_idx]):
                lb, ub = float(prob_interval[0]), float(prob_interval[1])
                if ub <= 0.0:
                    continue
                builder.add_next_value(
                    row,
                    int(s_next),
                    _to_interval(interval_cache, lb, ub),
                )

            p_abs_lb = float(imdp.P_absorbing[s_int][a_idx, 0])
            p_abs_ub = float(imdp.P_absorbing[s_int][a_idx, 1])
            if p_abs_ub > 0.0:
                builder.add_next_value(
                    row,
                    int(imdp.absorbing_state),
                    _to_interval(interval_cache, p_abs_lb, p_abs_ub),
                )

            row += 1

    builder.new_row_group(row)
    choice_labeling.add_label_to_choice(str(-1), row)
    builder.add_next_value(
        row,
        int(imdp.absorbing_state),
        interval_cache[(1.0, 1.0)],
    )

    matrix = builder.build()

    state_labeling = stormpy.storage.StateLabeling(imdp.nr_states)
    state_labeling.add_label("init")
    state_labeling.add_label_to_state("init", int(imdp.s_init))

    state_labeling.add_label("absorbing")
    state_labeling.add_label_to_state("absorbing", int(imdp.absorbing_state))

    state_labeling.add_label("critical")
    for s in imdp.states[critical_mask]:
        state_labeling.add_label_to_state("critical", int(s))

    state_labeling.add_label("goal")
    for s in imdp.states[goal_mask]:
        state_labeling.add_label_to_state("goal", int(s))

    components = stormpy.SparseIntervalModelComponents(
        transition_matrix=matrix,
        state_labeling=state_labeling,
    )
    components.choice_labeling = choice_labeling

    model = stormpy.storage.SparseIntervalMdp(components)
    return model


def RVI_STORM(
    args: argparse.Namespace,
    imdp: IMDP,
) -> Tuple[
    Float32[Array, "nr_states"],
    Bool,
    UInt8[Array, "nr_states"],
    Float32[Array, "nr_states p"],
]:
    """
    Robust value iteration for interval MDPs using Storm.

    Interface intentionally matches RVI_JAX as closely as possible.
    """

    start_time = time.time()

    model = _build_storm_imdp(imdp)

    logger.debug('%s', model)  # Print model info for debugging and verification

    prop = stormpy.parse_properties('Pmax=? [F "goal"]')[0]
    env = stormpy.Environment()
    env.solver_environment.minmax_solver_environment.method = (
        stormpy.MinMaxMethod.value_iteration
    )

    task = stormpy.CheckTask(prop.raw_formula, only_initial_states=False)
    task.set_produce_schedulers()
    if hasattr(task, "set_robust_uncertainty"):
        task.set_robust_uncertainty(True)
    elif hasattr(task, "set_uncertainty_resolution_mode"):
        task.set_uncertainty_resolution_mode(stormpy.UncertaintyResolutionMode.ROBUST)

    logger.info(f'- IDMP defined (took {time.time() - start_time:.3f}s); start robust dynamic programming...')

    result = stormpy.check_interval_mdp(model, task, env)

    float_dtype = getattr(args, "floatprecision", np.float32)
    V = np.asarray(result.get_values(), dtype=float_dtype)
    V = V[: imdp.nr_states]

    policy_labels = np.full(imdp.nr_states, fill_value=-1, dtype=np.int32)

    if result.has_scheduler:
        scheduler = result.scheduler
        for state in model.states:
            s = int(state)
            if s >= imdp.nr_states:
                continue

            choice = scheduler.get_choice(state)
            action_index = int(choice.get_deterministic_choice())
            action = state.actions[action_index]
            labels = list(action.labels)

            if labels:
                policy_labels[s] = int(labels[0])
    
    return V, policy_labels

