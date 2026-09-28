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
from .tube import _cells_from_flat_ids, _unique_id_chunks
from core.abstraction.partition import _compute_linear_strides

logger = logging.getLogger(__name__)

def _visited_rollout_cells(init_states, traces, was_done, env):
    """Collect visited cells using bounded arrays of integer IDs."""
    strides = _compute_linear_strides(env.number_per_dim)
    batch_size = max(1, 262144 // max(1, traces.shape[1]))
    steps = np.arange(traces.shape[1])

    def chunks():
        for start in range(0, len(init_states), batch_size):
            stop = start + batch_size
            lengths = np.sum(~was_done[start:stop], axis=1)
            # Include the terminal transition, exactly as tr[:sum(~done)].
            valid_traces = traces[start:stop][steps[None, :] < lengths[:, None]]
            states = np.concatenate((init_states[start:stop], valid_traces), axis=0)
            cells = np.clip(
                (states - env.obs_low) // env.bin_widths, 0, env.number_per_dim - 1
            ).astype(np.int64)
            yield cells @ strides

    return _cells_from_flat_ids(_unique_id_chunks(chunks()), env.number_per_dim)


class BaseRL:
    """Base class for reinforcement learning algorithms."""

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.params = None
        self._action_index_kernels = {}

    def train(self, seed: int = 0):
        raise NotImplementedError

    def get_predict_fn(self):
        raise NotImplementedError

    def _policy_apply(self, params, norm_obs):
        """Apply the deterministic policy to a batch of normalized observations."""
        raise NotImplementedError

    def predict_action(self, norm_obs: jnp.ndarray) -> jnp.ndarray:
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

    def _get_action_index_kernel(self, discrete_actions, num, batch_size):
        discrete_actions = np.ascontiguousarray(discrete_actions)
        key = (
            batch_size,
            num,
            discrete_actions.shape,
            discrete_actions.dtype.str,
            discrete_actions.tobytes(),
        )
        kernel = self._action_index_kernels.get(key)
        if kernel is not None:
            return kernel

        action_grid = jnp.asarray(discrete_actions)
        action_span = jnp.asarray(self.env.u_max - self.env.u_min)

        def distances_for_action(action):
            diff = (action - action_grid) / action_span
            return jnp.sum(jnp.square(diff), axis=-1)

        @jax.jit
        def extract_indices(params, state_batch):
            physical = self.env.obs_low_jnp + (state_batch + 0.5) * self.env.bin_widths_jnp
            norm_actions = self._policy_apply(params, self.env.normalize_obs(physical))
            actions = self.env.scale_action(norm_actions)
            distances = jax.vmap(distances_for_action)(actions)
            if num == 1:
                return jnp.argmin(distances, axis=1)[:, None]
            return jax.lax.top_k(-distances, num)[1]

        self._action_index_kernels[key] = extract_indices
        return extract_indices

    def get_policy_action_indices(self, states, discrete_actions, num=1, batch_size=8192):
        """Return nearest discrete-action indices using a fixed-shape JAX kernel."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        states_arr = np.atleast_2d(states)
        discrete_actions = np.asarray(discrete_actions)
        if len(discrete_actions) == 0:
            raise ValueError("discrete_actions must contain at least one action")
        num = len(range(len(discrete_actions))[:num])
        index_dtype = np.min_scalar_type(len(discrete_actions) - 1)
        selected = np.empty((len(states_arr), num), dtype=index_dtype)
        if len(states_arr) == 0 or num == 0:
            return selected

        kernel = self._get_action_index_kernel(discrete_actions, num, batch_size)

        for start in range(0, len(states_arr), batch_size):
            end = min(start + batch_size, len(states_arr))
            state_batch = states_arr[start:end]
            if len(state_batch) < batch_size:
                padded = np.empty((batch_size, *states_arr.shape[1:]), dtype=states_arr.dtype)
                padded[:len(state_batch)] = state_batch
                padded[len(state_batch):] = state_batch[0]
                state_batch = padded
            indices = np.asarray(kernel(self.params, jnp.asarray(state_batch)))
            selected[start:end] = indices[:end - start].astype(index_dtype, copy=False)

        return selected

    def get_policy_actions(self, states, discrete_actions, num=1, batch_size=8192):
        """Return nearest discrete actions; ties may select any equidistant action."""
        discrete_actions = np.asarray(discrete_actions)
        indices = self.get_policy_action_indices(states, discrete_actions, num=num, batch_size=batch_size)
        return discrete_actions[indices]

    def evaluate(self, args, discrete_actions=None, seed=0, output_dir=None, return_trajectories=False, return_visited_array=False):
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

            (_, _, final_goal), (trace, was_done) = jax.lax.scan(_step_body, (init_state, False, False), jax.random.split(rng_steps, self.cfg.rollout_steps))
            return init_state, trace, was_done, final_goal

        rngs = jax.random.split(jax.random.PRNGKey(seed), self.cfg.eval_episodes)
        init_states, traces, was_done, final_goals = [np.asarray(x) for x in jax.jit(jax.vmap(_single_rollout))(rngs)]

        visited_array = _visited_rollout_cells(init_states, traces, was_done, env)
        visited_cells = visited_array if return_visited_array else set(map(tuple, visited_array))
        trajectories = []
        for init_s, tr, done_m in zip(init_states[:100], traces[:100], was_done[:100]):
            full_tr = np.vstack([init_s[None, :], tr[:int(np.sum(~done_m))]])
            trajectories.append(full_tr)

        plot_dims = getattr(env.model, "plot_dimensions", None)
        if plot_dims is not None and len(plot_dims) == 2:
            plot_rl_trajectories(
                args, env.model, env, trajectories, list(plot_dims),
                Path(output_dir or getattr(self.cfg, "output_dir", "output")),
                algo_name=self.cfg.rl_algo,
            )

        result = (int(np.sum(final_goals)), visited_cells)
        return (*result, trajectories) if return_trajectories else result

    def _run_training(self, desc, runner_state, update_step, steps_per_update, num_chunks, format_fn):
        t_start = time()
        num_updates = self.cfg.total_timesteps // steps_per_update
        chunk_updates = max(num_updates // num_chunks, 1)
        num_chunks = max(1, num_updates // chunk_updates)
        total_steps = num_chunks * chunk_updates * steps_per_update

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
        pbar.close()
        logger.info("%s finished in %.2fs (%d timesteps).", desc, time() - t_start, total_steps)
        return runner_state
