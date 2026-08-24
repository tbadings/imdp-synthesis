import logging
from pathlib import Path
from time import time
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .config import RLConfig
from .env import JaxBenchmarkEnv, _sample_noise_jax, _in_boxes_jnp
from .models import ActorCritic, RunningMeanStd, normalize_obs
from .plotting import plot_rl_trajectories

logger = logging.getLogger(__name__)


# JAX-vectorized batch rollout evaluator compiled for maximum performance
def _build_batch_evaluator(actor_critic: ActorCritic, env: JaxBenchmarkEnv, max_steps: int):
    def _eval_rollouts(params, rms_obs, discrete_actions_jnp, rng_keys):
        def _single_rollout(rng):
            rng_reset, rng_steps = jax.random.split(rng)
            init_obs = jax.random.uniform(
                rng_reset,
                shape=env.test_reset_low_jnp.shape,
                minval=env.test_reset_low_jnp,
                maxval=env.test_reset_high_jnp,
            )

            step_keys = jax.random.split(rng_steps, max_steps)

            def _step_body(carry, key):
                curr_obs, is_done, hit_goal = carry

                cell = jnp.clip(
                    jnp.floor((curr_obs - env.obs_low_jnp) / env.bin_widths_jnp).astype(jnp.int32),
                    0,
                    env.number_per_dim_jnp - 1,
                )
                obs_q = env.obs_low_jnp + (cell.astype(jnp.float32) + 0.5) * env.bin_widths_jnp

                norm_obs = normalize_obs(rms_obs, obs_q)
                actor_mean, _, _ = actor_critic.apply(params, norm_obs)

                if discrete_actions_jnp is not None:
                    diff = actor_mean[None, :] - discrete_actions_jnp
                    dists = jnp.sum(jnp.square(diff), axis=-1)
                    action = discrete_actions_jnp[jnp.argmin(dists)]
                else:
                    action = actor_mean

                noise = _sample_noise_jax(env.model, key)
                next_obs = env.model.step(curr_obs, action, noise)

                in_goal = _in_boxes_jnp(next_obs, env.goal_jnp, inflate=0.0)
                in_critical = _in_boxes_jnp(next_obs, env.critical_jnp, inflate=0.0)
                out_of_bounds = jnp.any(next_obs < env.obs_low_jnp) | jnp.any(next_obs > env.obs_high_jnp)

                new_done = in_goal | in_critical | out_of_bounds
                next_is_done = is_done | new_done
                next_hit_goal = hit_goal | (in_goal & (~is_done))

                next_obs_masked = jnp.where(is_done, curr_obs, next_obs)
                step_out = (next_obs, is_done, new_done)
                next_carry = (next_obs_masked, next_is_done, next_hit_goal)
                return next_carry, step_out

            init_carry = (init_obs, jnp.array(False), jnp.array(False))
            (final_obs, final_done, final_goal), (next_obs_trace, was_done, is_term) = jax.lax.scan(
                _step_body, init_carry, step_keys
            )
            return init_obs, next_obs_trace, was_done, is_term, final_goal

        return jax.vmap(_single_rollout)(rng_keys)

    return jax.jit(_eval_rollouts)


# Rollout evaluation episodes under trained policy and record visited cells and trajectories
def evaluate_policy(
    actor_critic: ActorCritic,
    params,
    rms_obs: RunningMeanStd,
    base_model,
    env: JaxBenchmarkEnv,
    cfg: RLConfig,
    episodes: int,
    dims: Sequence[int],
    args,
    discrete_actions=None,
    seed: int = 0,
):
    logger.info(f"Running {episodes} evaluation rollouts in parallel (JAX)...")
    t = time()
    discrete_actions_jnp = jnp.asarray(discrete_actions, dtype=jnp.float32) if discrete_actions is not None else None

    # Compile and execute rollouts in parallel in JAX
    evaluator = _build_batch_evaluator(actor_critic, env, cfg.max_steps)
    rng = jax.random.PRNGKey(seed)
    rng_keys = jax.random.split(rng, episodes)

    init_obs_all, next_obs_trace_all, was_done_all, is_term_all, final_goal_all = evaluator(
        params, rms_obs, discrete_actions_jnp, rng_keys
    )

    init_obs_np = np.asarray(init_obs_all)
    next_obs_np = np.asarray(next_obs_trace_all)
    was_done_np = np.asarray(was_done_all)
    final_goal_np = np.asarray(final_goal_all)

    reached_goal = int(np.sum(final_goal_np))
    visited_cells = set()
    trajectories = []

    for i in range(episodes):
        # Steps that ran before the episode ended
        valid_steps = ~was_done_np[i]
        num_valid = int(np.sum(valid_steps))
        if num_valid > 0:
            ep_trace = np.vstack([init_obs_np[i:i+1], next_obs_np[i][:num_valid]])
        else:
            ep_trace = init_obs_np[i:i+1]

        cells = np.clip(
            np.floor((ep_trace - env.obs_low) / env.bin_widths).astype(int),
            0,
            env.number_per_dim - 1,
        )
        visited_cells.update(map(tuple, cells))
        trajectories.append(ep_trace)

    logger.info(f"- Evaluation rollouts completed in {time() - t:.2f} seconds.")
    t = time()

    plot_rl_trajectories(base_model, env, trajectories, dims, Path(getattr(args, "output_dir", "output")))
    logger.info(f"- Rollouts plotted completed in {time() - t:.2f} seconds.")

    total_cells = int(np.prod(base_model.partition["number_per_dim"]))
    return reached_goal, visited_cells, total_cells
