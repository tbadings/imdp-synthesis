from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from .config import RLConfig

class EnvState(NamedTuple):
    state: jnp.ndarray
    steps: jnp.ndarray
    prev_dist: jnp.ndarray

def _position_dims(model, obs_dim):
    """Dimensions used for the distance-to-goal term.

    Prefer an explicit ``pos_idx``, else the ``*_pos`` state variables (Drone models).
    Models without a position/velocity split fall back to the full state.
    """
    pos_idx = getattr(model, "pos_idx", None)
    if pos_idx is not None and len(pos_idx) > 0:
        return np.asarray(pos_idx, dtype=np.int64)
    names = getattr(model, "state_variables", None) or []
    idx = [i for i, name in enumerate(names) if str(name).endswith("_pos")]
    return np.asarray(idx if idx else range(obs_dim), dtype=np.int64)

class BenchmarkEnv:
    def __init__(self, model, cfg: RLConfig):
        self.model = model
        self.cfg = cfg
        self.obs_dim = model.n

        # State space and control bounds
        boundary = np.asarray(model.partition["boundary"], dtype=np.float32)
        self.obs_low, self.obs_high = boundary[0], boundary[1]
        self.u_min = np.asarray(model.uMin, dtype=np.float32)
        self.u_max = np.asarray(model.uMax, dtype=np.float32)

        # State space grid
        self.number_per_dim = np.asarray(model.partition["number_per_dim"], dtype=np.int64)
        self.bin_widths = (self.obs_high - self.obs_low) / self.number_per_dim

        # Goal & Critical sets
        self.goal = np.asarray(model.goal, dtype=np.float32)
        critical = getattr(model, "critical", None)
        self.critical = np.asarray(critical, dtype=np.float32) if critical is not None and len(critical) > 0 else np.empty((0, 2, self.obs_dim), dtype=np.float32)
        charging_station = getattr(model, "charging_station", None)
        self.charging_station = np.asarray(charging_station, dtype=np.float32) if charging_station is not None and len(charging_station) > 0 else np.empty((0, 2, self.obs_dim), dtype=np.float32)

        # Distance-to-goal metric: Euclidean distance from the position coordinates to the
        # goal *set* (zero inside it), normalised by the largest distance the domain admits.
        # Measuring to the set rather than to its centre keeps the term flat across the goal,
        # and dropping the velocity coordinates stops it from penalising travelling fast.
        self.position_dims = _position_dims(model, self.obs_dim)
        if len(self.goal):
            self._goal_lo = self.goal[0, 0][self.position_dims]
            self._goal_hi = self.goal[0, 1][self.position_dims]
        else:
            self._goal_lo = np.zeros(len(self.position_dims), dtype=np.float32)
            self._goal_hi = np.zeros(len(self.position_dims), dtype=np.float32)
        self._distance_scale = max(float(np.linalg.norm(np.maximum(
            self.obs_high[self.position_dims] - self._goal_hi,
            self._goal_lo - self.obs_low[self.position_dims],
        ))), 1e-6)

        # initial state sampling
        x0_cell = np.floor((model.x0 - self.obs_low) / self.bin_widths)
        cell_lb = self.obs_low + x0_cell * self.bin_widths
        eps = 0.1 * self.bin_widths
        self.reset_low = np.clip(cell_lb - eps, self.obs_low, self.obs_high)
        self.reset_high = np.clip(cell_lb + self.bin_widths + eps, self.obs_low, self.obs_high)

        # JAX tensors for fast JIT operations
        self.obs_low_jnp = jnp.asarray(self.obs_low)
        self.obs_high_jnp = jnp.asarray(self.obs_high)
        self.u_min_jnp = jnp.asarray(self.u_min)
        self.u_max_jnp = jnp.asarray(self.u_max)
        self.goal_jnp = jnp.asarray(self.goal)
        self.critical_jnp = jnp.asarray(self.critical)
        self.bin_widths_jnp = jnp.asarray(self.bin_widths)
        self.number_per_dim_jnp = jnp.asarray(self.number_per_dim, dtype=jnp.int32)
        self.reset_low_jnp = jnp.asarray(self.reset_low)
        self.reset_high_jnp = jnp.asarray(self.reset_high)
        self.position_dims_jnp = jnp.asarray(self.position_dims, dtype=jnp.int32)
        self.goal_lo_jnp = jnp.asarray(self._goal_lo)
        self.goal_hi_jnp = jnp.asarray(self._goal_hi)

    def distance_to_goal(self, state: jnp.ndarray) -> jnp.ndarray:
        """Normalised Euclidean distance from `state` to the goal set (0 inside, 1 at worst)."""
        pos = state[..., self.position_dims_jnp]
        return jnp.linalg.norm(pos - jnp.clip(pos, self.goal_lo_jnp, self.goal_hi_jnp), axis=-1) / self._distance_scale

def _sample_noise_jax(model, rng, shape=()):
    return model.noise.sample_jax(rng, shape=shape)

def _in_boxes_jnp(state: jnp.ndarray, boxes: jnp.ndarray, inflate: float = 0.0) -> jnp.ndarray:
    """Check if state(s) [..., D] fall inside any box [N, 2, D]. Supports batched shapes."""
    in_each = jnp.all((state[..., None, :] >= boxes[:, 0, :] - inflate) & 
                      (state[..., None, :] <= boxes[:, 1, :] + inflate), axis=-1)
    return jnp.any(in_each, axis=-1)

def _sample_safe_state(rng, env: BenchmarkEnv, num_candidates: int = 8) -> jnp.ndarray:
    """Sample a state in the domain outside critical and goal boxes."""
    candidates = jax.random.uniform(rng, shape=(num_candidates, env.obs_dim), minval=env.obs_low_jnp, maxval=env.obs_high_jnp)
    is_safe = ~(_in_boxes_jnp(candidates, env.critical_jnp) | _in_boxes_jnp(candidates, env.goal_jnp))
    return candidates[jnp.argmax(is_safe)]

def _env_step_jnp(rng, env_state: EnvState, action, env: BenchmarkEnv, noise_factor: float | None = None):
    action = jnp.clip(action, env.u_min_jnp, env.u_max_jnp)
    rng_noise, rng_reset = jax.random.split(rng)
    if noise_factor is None:
        noise_factor = env.cfg.train_noise_factor
    noise = noise_factor * _sample_noise_jax(env.model, rng_noise)
    next_state = env.model.step(env_state.state, action, noise)
    steps = env_state.steps + 1

    in_goal = _in_boxes_jnp(next_state, env.goal_jnp)
    in_critical = _in_boxes_jnp(next_state, env.critical_jnp, env.cfg.critical_margin)
    out_of_bounds = jnp.any(next_state < env.obs_low_jnp) | jnp.any(next_state > env.obs_high_jnp)

    # Euclidean distance-to-goal *cost* on surviving steps, alongside the flat per-step cost.
    dist = env.distance_to_goal(next_state)
    reward = jnp.select(
        [in_goal, in_critical, out_of_bounds],
        [env.cfg.goal_reward, env.cfg.unsafe_penalty, env.cfg.out_of_bounds_penalty],
        default=env.cfg.per_step_reward - env.cfg.distance_reward * dist,
    )

    terminated = in_goal | in_critical | out_of_bounds
    truncated = steps >= env.cfg.max_steps
    done = terminated | truncated

    # Reset upon termination / truncation in safe region
    reset_state = _sample_safe_state(rng_reset, env)
    next_env_state = EnvState(
        state=jnp.where(done, reset_state, next_state),
        steps=jnp.where(done, 0, steps),
        prev_dist=jnp.where(done, env.distance_to_goal(reset_state), dist),
    )

    info = {
        "in_goal": in_goal,
        "in_critical": in_critical,
        "out_of_bounds": out_of_bounds,
        "terminated": terminated,
        "truncated": truncated,
        # Successor *before* the auto-reset. The critic needs this to bootstrap correctly
        # when an episode ends on the step limit rather than on a real terminal state.
        "next_state": next_state,
    }
    return next_env_state.state, next_env_state, reward, done, info
