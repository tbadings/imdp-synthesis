from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from .config import RLConfig


class EnvState(NamedTuple):
    state: jnp.ndarray
    obs: jnp.ndarray
    steps: jnp.ndarray


def _in_boxes_jnp(state: jnp.ndarray, boxes: jnp.ndarray) -> jnp.ndarray:
    """Check if state(s) [..., D] fall inside any box [N, 2, D]. Vectorized and branch-free."""
    if boxes.shape[0] == 0:
        return jnp.zeros(state.shape[:-1], dtype=bool)
    in_each = jnp.all((state[..., None, :] >= boxes[:, 0, :]) & (state[..., None, :] <= boxes[:, 1, :]), axis=-1)
    return jnp.any(in_each, axis=-1)


class BenchmarkEnv:
    """Gymnax-compatible vectorized environment wrapper for benchmark models."""

    def __init__(self, model, cfg: RLConfig):
        self.model = model
        self.cfg = cfg
        self.obs_dim = model.n
        self.action_dim = len(model.uMin)

        # Domain bounds and grid setup
        boundary = np.asarray(model.partition["boundary"], dtype=np.float32)
        self.obs_low = boundary[0]
        self.obs_high = boundary[1]
        self.u_min = np.asarray(model.uMin, dtype=np.float32)
        self.u_max = np.asarray(model.uMax, dtype=np.float32)

        self.number_per_dim = np.asarray(model.partition["number_per_dim"], dtype=np.int64)
        self.bin_widths = (self.obs_high - self.obs_low) / self.number_per_dim

        # Specifications
        self.goal = np.asarray(getattr(model, "goal", []), dtype=np.float32).reshape(-1, 2, self.obs_dim)
        self.critical = np.asarray(getattr(model, "critical", []), dtype=np.float32).reshape(-1, 2, self.obs_dim)

        # Distance normalization span
        goal_lo = self.goal[0, 0] if len(self.goal) else np.zeros(self.obs_dim, dtype=np.float32)
        goal_hi = self.goal[0, 1] if len(self.goal) else np.zeros(self.obs_dim, dtype=np.float32)
        span = np.maximum(self.obs_high - goal_hi, goal_lo - self.obs_low)
        self.distance_span = np.where(span > 0, span, 1.0).astype(np.float32)
        self.distance_weights = np.broadcast_to(np.asarray(cfg.distance_cost, dtype=np.float32), (self.obs_dim,))

        # Reset region around initial state x0
        x0_cell = np.floor((model.x0 - self.obs_low) / self.bin_widths)
        cell_lb = self.obs_low + x0_cell * self.bin_widths
        eps = 0.1 * self.bin_widths
        self.reset_low = np.clip(cell_lb - eps, self.obs_low, self.obs_high)
        self.reset_high = np.clip(cell_lb + self.bin_widths + eps, self.obs_low, self.obs_high)

        # JAX arrays for JIT execution
        self.obs_low_jnp = jnp.asarray(self.obs_low)
        self.obs_high_jnp = jnp.asarray(self.obs_high)
        self.u_min_jnp = jnp.asarray(self.u_min)
        self.u_max_jnp = jnp.asarray(self.u_max)
        self.goal_jnp = jnp.asarray(self.goal)
        self.critical_jnp = jnp.asarray(self.critical)
        self.bin_widths_jnp = jnp.asarray(self.bin_widths)
        self.number_per_dim_jnp = jnp.asarray(self.number_per_dim, dtype=jnp.int32)
        self.goal_lo_jnp = jnp.asarray(goal_lo)
        self.goal_hi_jnp = jnp.asarray(goal_hi)
        self.distance_span_jnp = jnp.asarray(self.distance_span)
        self.distance_weights_jnp = jnp.asarray(self.distance_weights)

        self.reset_low_jnp = jnp.asarray(self.reset_low)
        self.reset_high_jnp = jnp.asarray(self.reset_high)

        prox_dims = cfg.proximity_dims or tuple(range(self.obs_dim))
        self.prox_dims_jnp = jnp.asarray(prox_dims, dtype=jnp.int32)
        self.obs_low_prox_jnp = self.obs_low_jnp[self.prox_dims_jnp]
        self.obs_high_prox_jnp = self.obs_high_jnp[self.prox_dims_jnp]
        if self.critical.shape[0] > 0:
            self.crit_low_prox_jnp = self.critical_jnp[:, 0, :][:, self.prox_dims_jnp]
            self.crit_high_prox_jnp = self.critical_jnp[:, 1, :][:, self.prox_dims_jnp]
        else:
            self.crit_low_prox_jnp = jnp.zeros((0, len(prox_dims)), dtype=jnp.float32)
            self.crit_high_prox_jnp = jnp.zeros((0, len(prox_dims)), dtype=jnp.float32)

    def normalize_obs(self, state: jnp.ndarray) -> jnp.ndarray:
        """Scale continuous state to [-1, 1]."""
        return 2.0 * (state - self.obs_low_jnp) / (self.obs_high_jnp - self.obs_low_jnp) - 1.0

    def scale_action(self, norm_action: jnp.ndarray) -> jnp.ndarray:
        """Scale normalized action [-1, 1] to [u_min, u_max]."""
        act_clipped = jnp.clip(norm_action, -1.0, 1.0)
        return self.u_min_jnp + 0.5 * (act_clipped + 1.0) * (self.u_max_jnp - self.u_min_jnp)

    def distance_to_goal(self, state: jnp.ndarray) -> jnp.ndarray:
        offset = state - jnp.clip(state, self.goal_lo_jnp, self.goal_hi_jnp)
        return jnp.linalg.norm(self.distance_weights_jnp * offset / self.distance_span_jnp, axis=-1)

    def min_distance_to_boundary(self, state: jnp.ndarray) -> jnp.ndarray:
        """Compute minimum Euclidean distance to boundaries and obstacles over proximity_dims."""
        s = state[..., self.prox_dims_jnp]
        d_b = jnp.min(jnp.minimum(s - self.obs_low_prox_jnp, self.obs_high_prox_jnp - s), axis=-1)
        d_b = jnp.maximum(d_b, 0.0)
        if self.critical.shape[0] == 0:
            return d_b
        s_exp = jnp.expand_dims(s, axis=-2)
        delta = jnp.maximum(0.0, jnp.maximum(self.crit_low_prox_jnp - s_exp, s_exp - self.crit_high_prox_jnp))
        crit_dist = jnp.min(jnp.linalg.norm(delta, axis=-1), axis=-1)
        return jnp.minimum(d_b, crit_dist)


    def reset(self, rng_batch: jax.Array) -> tuple[jnp.ndarray, EnvState]:
        """Vectorized reset over parallel environments."""
        def _single_reset(rng):
            state = jax.random.uniform(rng, (self.obs_dim,), minval=self.obs_low_jnp, maxval=self.obs_high_jnp)
            norm_obs = self.normalize_obs(state)
            return norm_obs, EnvState(state=state, obs=norm_obs, steps=jnp.array(0, dtype=jnp.int32))

        return jax.vmap(_single_reset)(rng_batch)

    def step(self, rng_batch, env_state: EnvState, action: jnp.ndarray):
        """Vectorized step over parallel environments."""
        def _single_step(rng, s, norm_act):
            physical_act = self.scale_action(norm_act)
            rng_noise, rng_reset = jax.random.split(rng)
            noise = self.model.noise.sample_jax(rng_noise)
            next_state = self.model.step(s.state, physical_act, noise)
            steps = s.steps + 1

            in_goal = _in_boxes_jnp(next_state, self.goal_jnp)
            in_critical = _in_boxes_jnp(next_state, self.critical_jnp)
            out_of_bounds = jnp.any(next_state < self.obs_low_jnp) | jnp.any(next_state > self.obs_high_jnp)

            dist = self.distance_to_goal(next_state)
            min_dist = self.min_distance_to_boundary(next_state)
            proximity_penalty = self.cfg.proximity_penalty * jnp.maximum(1.0 - min_dist, 0.0)

            reward = jnp.select(
                [in_goal, in_critical, out_of_bounds],
                [self.cfg.goal_reward, self.cfg.unsafe_penalty, self.cfg.out_of_bounds_penalty],
                default=-self.cfg.per_step_cost - dist - proximity_penalty,
            )

            terminated = in_goal | in_critical | out_of_bounds
            truncated = steps >= self.cfg.max_steps
            done = terminated | truncated

            reset_state = jax.random.uniform(rng_reset, (self.obs_dim,), minval=self.obs_low_jnp, maxval=self.obs_high_jnp)
            final_state = jnp.where(done, reset_state, next_state)
            final_steps = jnp.where(done, 0, steps)
            final_obs = self.normalize_obs(final_state)

            next_s = EnvState(state=final_state, obs=final_obs, steps=final_steps)
            info = {
                "terminated": terminated,
                "next_obs": self.normalize_obs(next_state),
            }
            return final_obs, next_s, reward, done, info

        return jax.vmap(_single_step)(rng_batch, env_state, action)
