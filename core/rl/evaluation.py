import logging
from pathlib import Path
from time import time
from typing import Sequence
import jax
import jax.numpy as jnp
import numpy as np

from .config import RLConfig
from .env import BenchmarkEnv, _sample_noise_jax, _in_boxes_jnp
from .policy import ActorCritic
from .plotting import plot_rl_trajectories

logger = logging.getLogger(__name__)

def _build_batch_evaluator(actor_critic: ActorCritic, env: BenchmarkEnv, max_steps: int):
    def _single_rollout(params, discrete_actions_jnp, rng):
        rng_reset, rng_steps = jax.random.split(rng)
        init_obs = jax.random.uniform(rng_reset, shape=env.reset_low_jnp.shape, minval=env.reset_low_jnp, maxval=env.reset_high_jnp)
        def _step_body(carry, key):
            curr_obs, is_done, hit_goal = carry

            # Quantize observation to grid cell center
            cell = jnp.clip((curr_obs - env.obs_low_jnp) // env.bin_widths_jnp, 0, env.number_per_dim_jnp - 1)
            obs_q = env.obs_low_jnp + (cell + 0.5) * env.bin_widths_jnp

            actor_mean = actor_critic.apply(params, obs_q)[0]
            if discrete_actions_jnp is not None:
                action = discrete_actions_jnp[jnp.argmin(jnp.sum((actor_mean - discrete_actions_jnp) ** 2, axis=-1))]
            else:
                action = actor_mean

            next_obs = env.model.step(curr_obs, action, _sample_noise_jax(env.model, key))
            in_goal = _in_boxes_jnp(next_obs, env.goal_jnp)
            terminal = in_goal | _in_boxes_jnp(next_obs, env.critical_jnp) | jnp.any((next_obs < env.obs_low_jnp) | (next_obs > env.obs_high_jnp))

            next_carry = (jnp.where(is_done, curr_obs, next_obs), is_done | terminal, hit_goal | (in_goal & ~is_done))
            return next_carry, (next_obs, is_done)

        (_, _, final_goal), (next_obs_trace, was_done) = jax.lax.scan(
            _step_body, (init_obs, False, False), jax.random.split(rng_steps, max_steps)
        )
        return init_obs, next_obs_trace, was_done, final_goal

    return jax.jit(jax.vmap(_single_rollout, in_axes=(None, None, 0)))

def evaluate_policy(
    actor_critic: ActorCritic,
    params,
    base_model,
    env: BenchmarkEnv,
    cfg: RLConfig,
    episodes: int,
    dims: Sequence[int],
    args,
    discrete_actions=None,
    seed: int = 0,
):
    logger.info(f"Running {episodes} evaluation rollouts in parallel (JAX)...")
    t0 = time()

    evaluator = _build_batch_evaluator(actor_critic, env, cfg.max_steps)
    discrete_actions_jnp = jnp.asarray(discrete_actions, dtype=jnp.float32) if discrete_actions is not None else None
    rng_keys = jax.random.split(jax.random.PRNGKey(seed), episodes)

    init_obs_all, next_obs_all, was_done_all, final_goal_all = [
        np.asarray(x) for x in evaluator(params, discrete_actions_jnp, rng_keys)
    ]

    visited_cells, trajectories = set(), []
    for init_obs, ep_next, done_mask in zip(init_obs_all, next_obs_all, was_done_all):
        trace = np.vstack([init_obs[None, :], ep_next[:int(np.sum(~done_mask))]])
        cells = np.clip((trace - env.obs_low) // env.bin_widths, 0, env.number_per_dim - 1).astype(int)
        visited_cells.update(map(tuple, cells))
        trajectories.append(trace)

    logger.info(f"- Evaluation rollouts completed in {time() - t0:.2f} seconds.")
    t0 = time()
    plot_rl_trajectories(base_model, env, trajectories, dims, Path(getattr(args, "output_dir", "output")))
    logger.info(f"- Rollouts plotted completed in {time() - t0:.2f} seconds.")

    return int(np.sum(final_goal_all)), visited_cells, int(np.prod(base_model.partition["number_per_dim"]))
