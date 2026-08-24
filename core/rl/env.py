from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from .config import RLConfig


class EnvState(NamedTuple):
    state: jnp.ndarray
    steps: jnp.ndarray
    prev_dist: jnp.ndarray


class JaxBenchmarkEnv:
    def __init__(self, model, cfg: RLConfig):
        self.model = model
        self.cfg = cfg
        self.obs_dim = model.n

        # Boundaries and action limits
        boundary = np.asarray(model.partition["boundary"], dtype=np.float32)
        self.obs_low, self.obs_high = boundary[0], boundary[1]
        self.u_min = np.asarray(model.uMin, dtype=np.float32)
        self.u_max = np.asarray(model.uMax, dtype=np.float32)

        # Discretization grid
        self.number_per_dim = np.asarray(model.partition["number_per_dim"], dtype=np.int64)
        self.bin_widths = (self.obs_high - self.obs_low) / self.number_per_dim

        # Goal & Critical sets
        self.goal = np.asarray(getattr(model, "goal", np.empty((0, 2, self.obs_dim))), dtype=np.float32)
        self.critical = np.asarray(getattr(model, "critical", np.empty((0, 2, self.obs_dim))), dtype=np.float32)
        self._goal_center = 0.5 * (self.goal[0, 0] + self.goal[0, 1]) if len(self.goal) else np.zeros(self.obs_dim, dtype=np.float32)
        self._domain_scale = max(float(np.linalg.norm(self.obs_high - self.obs_low)), 1e-6)

        # Reset bounds for initial state sampling
        infl = np.asarray(getattr(model, "reset_inflation", None) or [(0, 0)] * self.obs_dim, dtype=np.float32)
        x0_cell = np.clip(np.floor((model.x0 - self.obs_low) / self.bin_widths), 0, self.number_per_dim - 1)
        cell_lb = self.obs_low + x0_cell * self.bin_widths
        eps = 0.1 * self.bin_widths
        self.test_reset_low = np.clip(cell_lb + infl[:, 0] * self.bin_widths - eps, self.obs_low, self.obs_high)
        self.test_reset_high = np.clip(cell_lb + self.bin_widths + infl[:, 1] * self.bin_widths + eps, self.obs_low, self.obs_high)

        # JAX tensors for fast JIT operations
        self.obs_low_jnp = jnp.asarray(self.obs_low)
        self.obs_high_jnp = jnp.asarray(self.obs_high)
        self.u_min_jnp = jnp.asarray(self.u_min)
        self.u_max_jnp = jnp.asarray(self.u_max)
        self.goal_jnp = jnp.asarray(self.goal)
        self.critical_jnp = jnp.asarray(self.critical)
        self.bin_widths_jnp = jnp.asarray(self.bin_widths)
        self.number_per_dim_jnp = jnp.asarray(self.number_per_dim, dtype=jnp.int32)
        self.goal_center_jnp = jnp.asarray(self._goal_center)
        self.test_reset_low_jnp = jnp.asarray(self.test_reset_low)
        self.test_reset_high_jnp = jnp.asarray(self.test_reset_high)


def _sample_noise_jax(model, rng, shape=()):
    if hasattr(model.noise, "sample_jax"):
        return model.noise.sample_jax(rng, shape=shape)
    if isinstance(model.noise, dict) and "stdev" in model.noise:
        stdev = jnp.asarray(model.noise["stdev"], dtype=jnp.float32)
        mean = jnp.asarray(model.noise.get("mean", 0.0), dtype=jnp.float32)
        s_shape = (shape if isinstance(shape, tuple) else (shape,)) + (stdev.shape[0],)
        return mean + stdev * jax.random.normal(rng, shape=s_shape)
    return jnp.zeros((shape if isinstance(shape, tuple) else (shape,)) + (model.n,), dtype=jnp.float32)


def _in_boxes_jnp(state: jnp.ndarray, boxes: jnp.ndarray, inflate: float = 0.0) -> jnp.ndarray:
    """Check if state(s) [..., D] fall inside any box [N, 2, D]. Supports batched shapes."""
    if boxes.shape[0] == 0:
        return jnp.zeros(state.shape[:-1], dtype=bool) if state.ndim > 1 else jnp.array(False)
    in_each = jnp.all((state[..., None, :] >= boxes[:, 0, :] - inflate) & 
                      (state[..., None, :] <= boxes[:, 1, :] + inflate), axis=-1)
    return jnp.any(in_each, axis=-1)


def _sample_safe_state(rng, env: JaxBenchmarkEnv, num_candidates: int = 16) -> jnp.ndarray:
    """Sample a state in the domain outside critical and goal boxes."""
    candidates = jax.random.uniform(rng, shape=(num_candidates, env.obs_dim), minval=env.obs_low_jnp, maxval=env.obs_high_jnp)
    is_safe = ~(_in_boxes_jnp(candidates, env.critical_jnp) | _in_boxes_jnp(candidates, env.goal_jnp))
    return candidates[jnp.argmax(is_safe)]


def _env_step_jnp(rng, env_state: EnvState, action, env: JaxBenchmarkEnv, noise_factor: float = 2.0):
    action = jnp.clip(action, env.u_min_jnp, env.u_max_jnp)
    rng_noise, rng_reset = jax.random.split(rng)
    noise = noise_factor * _sample_noise_jax(env.model, rng_noise)
    next_state = env.model.step(env_state.state, action, noise)
    steps = env_state.steps + 1

    in_goal = _in_boxes_jnp(next_state, env.goal_jnp)
    in_critical = _in_boxes_jnp(next_state, env.critical_jnp)
    out_of_bounds = jnp.any(next_state < env.obs_low_jnp) | jnp.any(next_state > env.obs_high_jnp)

    dist = jnp.linalg.norm(next_state - env.goal_center_jnp)
    dist_rew = env.cfg.distance_reward * (env_state.prev_dist - dist) / env._domain_scale

    reward = jnp.select(
        [in_goal, in_critical, out_of_bounds],
        [env.cfg.goal_reward, env.cfg.unsafe_penalty, env.cfg.out_of_bounds_penalty],
        default=dist_rew + env.cfg.per_step_reward,
    )

    terminated = in_goal | in_critical | out_of_bounds
    truncated = steps >= env.cfg.max_steps
    done = terminated | truncated

    # Auto-reset upon termination / truncation in safe region
    reset_state = _sample_safe_state(rng_reset, env)
    next_env_state = EnvState(
        state=jnp.where(done, reset_state, next_state),
        steps=jnp.where(done, 0, steps),
        prev_dist=jnp.where(done, jnp.linalg.norm(reset_state - env.goal_center_jnp), dist),
    )

    info = {
        "in_goal": in_goal,
        "in_critical": in_critical,
        "out_of_bounds": out_of_bounds,
        "terminated": terminated,
        "truncated": truncated,
    }
    return next_state, next_env_state, reward, done, info
