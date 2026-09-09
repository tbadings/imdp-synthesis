import logging
from pathlib import Path
import pickle
from time import time
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .env import _in_boxes_jnp
from .plotting import plot_rl_trajectories

logger = logging.getLogger(__name__)


class BaseRL:
    """Base class for reinforcement learning algorithms."""

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.params = None

    def train(self, seed: int = 0):
        raise NotImplementedError

    def get_predict_fn(self):
        raise NotImplementedError

    def predict_action(self, norm_obs: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        return self.get_predict_fn()(norm_obs)

    def save(self, filepath):
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"params": jax.tree_util.tree_map(np.asarray, self.params)}, f)

    def load(self, filepath):
        p = Path(filepath)
        p = p / "rl_policy.pkl" if p.is_dir() else p
        with open(p, "rb") as f:
            saved = pickle.load(f)
        self.params = saved.get("params", saved) if isinstance(saved, dict) else saved
        return self

    def get_policy_actions(self, states, discrete_actions, num=1):
        states_arr = np.atleast_2d(states)
        physical = self.env.obs_low + (states_arr + 0.5) * self.env.bin_widths
        actions = np.asarray(self.env.scale_action(self.predict_action(self.env.normalize_obs(physical))))
        diff = (actions[:, None, :] - discrete_actions[None, :, :]) / (self.env.u_max - self.env.u_min)
        top_k = np.argsort(np.sum(diff ** 2, axis=-1), axis=1)[:, :num]
        return discrete_actions[top_k], discrete_actions[top_k[:, 0]]

    def evaluate(self, discrete_actions=None, seed=0, output_dir=None):
        env = self.env
        predict_fn = self.get_predict_fn()
        action_span = env.u_max_jnp - env.u_min_jnp
        disc_actions_jnp = jnp.asarray(discrete_actions, dtype=jnp.float32) if discrete_actions is not None else None

        def _single_rollout(rng):
            rng_init, rng_steps = jax.random.split(rng)
            init_state = jax.random.uniform(rng_init, (env.obs_dim,), minval=env.reset_low_jnp, maxval=env.reset_high_jnp)

            def _step_body(carry, key):
                curr_state, is_done, hit_goal = carry
                cell = jnp.clip((curr_state - env.obs_low_jnp) // env.bin_widths_jnp, 0, env.number_per_dim_jnp - 1)
                norm_act = predict_fn(env.normalize_obs(env.obs_low_jnp + (cell + 0.5) * env.bin_widths_jnp))
                cont_act = env.scale_action(norm_act)
                act = disc_actions_jnp[jnp.argmin(jnp.sum(((cont_act - disc_actions_jnp) / action_span) ** 2, axis=-1))] if disc_actions_jnp is not None else cont_act
                next_state = env.model.step(curr_state, act, env.model.noise.sample_jax(key))
                in_goal = _in_boxes_jnp(next_state, env.goal_jnp)
                terminal = in_goal | _in_boxes_jnp(next_state, env.critical_jnp) | jnp.any((next_state < env.obs_low_jnp) | (next_state > env.obs_high_jnp))
                return (jnp.where(is_done, curr_state, next_state), is_done | terminal, hit_goal | (in_goal & ~is_done)), (next_state, is_done)

            (final_state, _, final_goal), (trace, was_done) = jax.lax.scan(_step_body, (init_state, False, False), jax.random.split(rng_steps, self.cfg.rollout_steps))
            return init_state, trace, was_done, final_goal

        rngs = jax.random.split(jax.random.PRNGKey(seed), self.cfg.eval_episodes)
        init_states, traces, was_done, final_goals = [np.asarray(x) for x in jax.jit(jax.vmap(_single_rollout))(rngs)]

        visited_cells, trajectories = set(), []
        for init_s, tr, done_m in zip(init_states, traces, was_done):
            full_tr = np.vstack([init_s[None, :], tr[:int(np.sum(~done_m))]])
            cells = np.clip((full_tr - env.obs_low) // env.bin_widths, 0, env.number_per_dim - 1).astype(int)
            visited_cells.update(map(tuple, cells))
            trajectories.append(full_tr)

        plot_dims = getattr(env.model, "plot_dimensions", None)
        if plot_dims is not None and len(plot_dims) == 2:
            plot_rl_trajectories(
                env.model, env, trajectories, list(plot_dims),
                Path(output_dir or getattr(self.cfg, "output_dir", "output")),
                algo_name=self.cfg.rl_algo,
            )

        return int(np.sum(final_goals)), visited_cells, int(np.prod(env.number_per_dim))

    def _run_training(self, desc, runner_state, update_step, steps_per_update, num_chunks, format_fn):
        t_start = time()
        num_updates = self.cfg.total_timesteps // steps_per_update
        chunk_updates = max(num_updates // num_chunks, 1)
        total_steps = num_updates * steps_per_update

        @jax.jit
        def run_chunk(state):
            return jax.lax.scan(update_step, state, None, length=chunk_updates)

        pbar = tqdm(total=total_steps, desc=desc, unit="step")
        smooth_rew = None
        for _ in range(num_chunks):
            runner_state, chunk_metrics = run_chunk(runner_state)
            mean_metrics = {k: float(np.mean(v)) for k, v in chunk_metrics.items()}
            mean_rew = mean_metrics["reward"]
            smooth_rew = mean_rew if smooth_rew is None else 0.95 * smooth_rew + 0.05 * mean_rew
            pbar.set_postfix(format_fn(mean_metrics, smooth_rew))
            pbar.update(chunk_updates * steps_per_update)
            print(flush=True)
        pbar.close()
        logger.info("%s finished in %.2fs (%d timesteps).", desc, time() - t_start, total_steps)
        return runner_state
