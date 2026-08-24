import itertools
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import time
from typing import NamedTuple, Sequence, Tuple, Dict, Any, Optional

import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
import optax
from tqdm import tqdm

from core.abstraction.partition import _compute_linear_strides

logger = logging.getLogger(__name__)

# Max neighbor indices processed per chunk to bound peak memory during grid inflation
INFLATE_IDS_PER_BATCH = 1 << 23
# Batch chunk size for forward reachable set expansions
CHUNK_SIZE = 16384


# Configuration dataclass holding reward weights and horizon limits for RL training
@dataclass
class RLConfig:
    max_steps: int
    goal_reward: float
    unsafe_penalty: float
    out_of_bounds_penalty: float
    revisit_penalty: float
    distance_reward: float = 0.0
    per_step_reward: float = 0.0


# Stateless environment definition for pure JAX execution
class EnvState(NamedTuple):
    state: jnp.ndarray
    steps: jnp.ndarray
    prev_dist: jnp.ndarray


class JaxBenchmarkEnv:
    def __init__(self, model, cfg: RLConfig, previous_cells=None):
        self.model = model
        self.cfg = cfg

        boundary = np.asarray(self.model.partition["boundary"], dtype=np.float32)
        self.obs_low = boundary[0]
        self.obs_high = boundary[1]
        self.u_min = np.asarray(self.model.uMin, dtype=np.float32)
        self.u_max = np.asarray(self.model.uMax, dtype=np.float32)

        self.number_per_dim = np.asarray(self.model.partition["number_per_dim"], dtype=np.int64)
        self.bin_widths = (self.obs_high - self.obs_low) / self.number_per_dim
        self.strides = _compute_linear_strides(self.number_per_dim)

        self.goal = np.asarray(getattr(self.model, "goal", np.empty((0, 2, self.model.n))), dtype=np.float32)
        self.critical = np.asarray(getattr(self.model, "critical", np.empty((0, 2, self.model.n))), dtype=np.float32)

        self.previous_cells = set() if previous_cells is None else set(previous_cells)

        self._goal_center = (
            0.5 * (self.goal[0, 0] + self.goal[0, 1])
            if self.goal.size > 0
            else np.zeros(self.model.n, dtype=np.float32)
        )
        self._domain_scale = max(float(np.linalg.norm(self.obs_high - self.obs_low)), 1e-6)

        # Reset bounds for initial state evaluation
        cell = np.asarray(self.state_to_cell(self.model.x0), dtype=np.float32)
        cell_lb = self.obs_low + cell * self.bin_widths
        cell_ub = cell_lb + self.bin_widths
        reset_infl = getattr(self.model, "reset_inflation", None) or ([(0, 0)] * self.model.n)
        lo = np.array([l for l, h in reset_infl], dtype=np.float32)
        hi = np.array([h for l, h in reset_infl], dtype=np.float32)
        eps = 0.1 * self.bin_widths
        self.test_reset_low = np.clip(cell_lb + lo * self.bin_widths - eps, self.obs_low, self.obs_high)
        self.test_reset_high = np.clip(cell_ub + hi * self.bin_widths + eps, self.obs_low, self.obs_high)

        # JAX arrays for fast JIT operations
        self.obs_low_jnp = jnp.asarray(self.obs_low, dtype=jnp.float32)
        self.obs_high_jnp = jnp.asarray(self.obs_high, dtype=jnp.float32)
        self.u_min_jnp = jnp.asarray(self.u_min, dtype=jnp.float32)
        self.u_max_jnp = jnp.asarray(self.u_max, dtype=jnp.float32)
        self.goal_jnp = jnp.asarray(self.goal, dtype=jnp.float32)
        self.critical_jnp = jnp.asarray(self.critical, dtype=jnp.float32)
        self.bin_widths_jnp = jnp.asarray(self.bin_widths, dtype=jnp.float32)
        self.number_per_dim_jnp = jnp.asarray(self.number_per_dim, dtype=jnp.int32)
        self.strides_jnp = jnp.asarray(self.strides, dtype=jnp.int32)
        self.goal_center_jnp = jnp.asarray(self._goal_center, dtype=jnp.float32)
        self.test_reset_low_jnp = jnp.asarray(self.test_reset_low, dtype=jnp.float32)
        self.test_reset_high_jnp = jnp.asarray(self.test_reset_high, dtype=jnp.float32)

    def state_to_cell(self, obs):
        indices = np.floor((np.asarray(obs, dtype=np.float64) - self.obs_low) / self.bin_widths).astype(int)
        return tuple(np.clip(indices, 0, self.number_per_dim - 1).tolist())

    def cell_to_center(self, cell):
        return self.obs_low + (np.asarray(cell, dtype=np.float32) + 0.5) * self.bin_widths

    @staticmethod
    def _in_boxes_np(state, boxes, inflate=0.0):
        if boxes.size == 0:
            return False
        return bool(np.any(np.all((state >= boxes[:, 0, :] - inflate) & (state <= boxes[:, 1, :] + inflate), axis=1)))


def _sample_noise_jax(model, rng, shape=()):
    if hasattr(model.noise, "sample_jax"):
        return model.noise.sample_jax(rng, shape=shape)
    elif isinstance(model.noise, dict) and "stdev" in model.noise:
        stdev = jnp.asarray(model.noise["stdev"], dtype=jnp.float32)
        mean = jnp.asarray(model.noise.get("mean", 0.0), dtype=jnp.float32)
        if isinstance(shape, int):
            shape = (shape,)
        s_shape = tuple(shape) + (stdev.shape[0],)
        return mean + stdev * jax.random.normal(rng, shape=s_shape)
    else:
        return jnp.zeros(tuple(shape) + (model.n,), dtype=jnp.float32)


def _in_boxes_jnp(state, boxes, inflate=0.0):
    if boxes.shape[0] == 0:
        return jnp.array(False)
    in_each = jnp.all((state >= boxes[:, 0, :] - inflate) & (state <= boxes[:, 1, :] + inflate), axis=-1)
    return jnp.any(in_each)


def _sample_safe_state(rng, env: JaxBenchmarkEnv, num_candidates: int = 16):
    """Sample a state that is guaranteed to be in the safe domain (not inside critical or goal regions)."""
    candidates = jax.random.uniform(
        rng,
        shape=(num_candidates, env.obs_low_jnp.shape[0]),
        minval=env.obs_low_jnp,
        maxval=env.obs_high_jnp,
    )
    if env.critical_jnp.shape[0] > 0:
        in_crit = jnp.any(
            jnp.all(
                (candidates[:, None, :] >= env.critical_jnp[None, :, 0, :])
                & (candidates[:, None, :] <= env.critical_jnp[None, :, 1, :]),
                axis=-1,
            ),
            axis=-1,
        )
    else:
        in_crit = jnp.zeros(num_candidates, dtype=bool)

    if env.goal_jnp.shape[0] > 0:
        in_g = jnp.any(
            jnp.all(
                (candidates[:, None, :] >= env.goal_jnp[None, :, 0, :])
                & (candidates[:, None, :] <= env.goal_jnp[None, :, 1, :]),
                axis=-1,
            ),
            axis=-1,
        )
    else:
        in_g = jnp.zeros(num_candidates, dtype=bool)

    is_safe = (~in_crit) & (~in_g)
    safe_idx = jnp.argmax(is_safe)
    return candidates[safe_idx]


def _env_step_jnp(rng, env_state: EnvState, action, env: JaxBenchmarkEnv, revisit_mask=None, noise_factor=2.0):
    action = jnp.clip(action, env.u_min_jnp, env.u_max_jnp)
    rng_noise, rng_reset = jax.random.split(rng)
    noise = noise_factor * _sample_noise_jax(env.model, rng_noise)
    next_state = env.model.step(env_state.state, action, noise)
    steps = env_state.steps + 1

    in_goal = _in_boxes_jnp(next_state, env.goal_jnp, inflate=0.0)
    in_critical = _in_boxes_jnp(next_state, env.critical_jnp, inflate=0.0)
    out_of_bounds = jnp.any(next_state < env.obs_low_jnp) | jnp.any(next_state > env.obs_high_jnp)

    dist = jnp.linalg.norm(next_state - env.goal_center_jnp)
    dist_rew = env.cfg.distance_reward * (env_state.prev_dist - dist) / env._domain_scale

    base_reward = jnp.where(
        in_goal,
        env.cfg.goal_reward,
        jnp.where(
            in_critical,
            env.cfg.unsafe_penalty,
            jnp.where(
                out_of_bounds,
                env.cfg.out_of_bounds_penalty,
                dist_rew + env.cfg.per_step_reward,
            ),
        ),
    )

    cell = jnp.clip(
        jnp.floor((next_state - env.obs_low_jnp) / env.bin_widths_jnp).astype(jnp.int32),
        0,
        env.number_per_dim_jnp - 1,
    )
    flat_cell = jnp.sum(cell * env.strides_jnp)

    if revisit_mask is not None:
        is_revisit = revisit_mask[flat_cell]
        reward = jnp.where(
            is_revisit & (~in_goal) & (~in_critical) & (~out_of_bounds),
            base_reward - env.cfg.revisit_penalty,
            base_reward,
        )
    else:
        reward = base_reward

    terminated = in_goal | in_critical | out_of_bounds
    truncated = steps >= env.cfg.max_steps
    done = terminated | truncated

    # Auto-reset upon termination / truncation in safe area
    reset_state = _sample_safe_state(rng_reset, env)
    reset_dist = jnp.linalg.norm(reset_state - env.goal_center_jnp)

    next_env_state = EnvState(
        state=jnp.where(done, reset_state, next_state),
        steps=jnp.where(done, 0, steps),
        prev_dist=jnp.where(done, reset_dist, dist),
    )

    info = {
        "in_goal": in_goal,
        "in_critical": in_critical,
        "out_of_bounds": out_of_bounds,
        "terminated": terminated,
        "truncated": truncated,
        "flat_cell": flat_cell,
    }
    return next_state, next_env_state, reward, done, info


# Neural Network Policy & Value Function Architecture in Flax
class ActorCritic(nn.Module):
    action_dim: int
    pi_arch: Sequence[int] = (128, 128)
    vf_arch: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, x):
        # Policy / Actor network
        actor_x = x
        for h in self.pi_arch:
            actor_x = nn.relu(
                nn.Dense(h, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.zeros)(actor_x)
            )
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=nn.initializers.orthogonal(0.01), bias_init=nn.initializers.zeros
        )(actor_x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))

        # Critic / Value network
        critic_x = x
        for h in self.vf_arch:
            critic_x = nn.relu(
                nn.Dense(h, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.zeros)(critic_x)
            )
        critic_val = nn.Dense(
            1, kernel_init=nn.initializers.orthogonal(1.0), bias_init=nn.initializers.zeros
        )(critic_x)

        return actor_mean, log_std, jnp.squeeze(critic_val, axis=-1)


# Running mean / std statistics for observation normalization
class RunningMeanStd(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def init_running_mean_std(shape):
    return RunningMeanStd(
        mean=jnp.zeros(shape, dtype=jnp.float32),
        var=jnp.ones(shape, dtype=jnp.float32),
        count=jnp.array(1e-4, dtype=jnp.float32),
    )


def update_running_mean_std(rms: RunningMeanStd, batch: jnp.ndarray):
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = float(batch.shape[0])

    delta = batch_mean - rms.mean
    total_count = rms.count + batch_count

    new_mean = rms.mean + delta * (batch_count / total_count)
    m_a = rms.var * rms.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * (rms.count * batch_count / total_count)
    new_var = m2 / total_count

    return RunningMeanStd(mean=new_mean, var=new_var, count=total_count)


def normalize_obs(rms: RunningMeanStd, obs: jnp.ndarray, clip=10.0):
    normed = (obs - rms.mean) / jnp.sqrt(rms.var + 1e-8)
    return jnp.clip(normed, -clip, clip)


# Gaussian action sampling and log probability math
def gaussian_sample(rng, mean, log_std):
    std = jnp.exp(log_std)
    return mean + std * jax.random.normal(rng, shape=mean.shape)


def gaussian_log_prob(action, mean, log_std):
    std = jnp.exp(log_std)
    var = jnp.square(std)
    log_scale = log_std + 0.5 * jnp.log(2.0 * jnp.pi)
    return -0.5 * jnp.sum(jnp.square(action - mean) / var + 2.0 * log_scale, axis=-1)


def gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)), axis=-1)


class Transition(NamedTuple):
    obs: jnp.ndarray
    raw_obs: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray


class FlatTransition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    advantage: jnp.ndarray
    target: jnp.ndarray


# PureJaxRL PPO continuous action training pipeline
def train_ppo(
    env: JaxBenchmarkEnv,
    args,
    pi_arch: Sequence[int],
    vf_arch: Sequence[int],
    previous_cells=None,
    seed: int = 0,
):
    rng = jax.random.PRNGKey(seed)
    n_envs = args.n_envs
    n_steps = args.n_steps
    rl_batch_size = args.rl_batch_size
    learning_rate = args.learning_rate
    ent_coef = args.ent_coef
    total_timesteps = args.total_timesteps

    batch_size = n_envs * n_steps
    num_minibatches = max(1, batch_size // rl_batch_size)
    minibatch_size = batch_size // num_minibatches
    num_updates = max(1, total_timesteps // batch_size)
    update_epochs = 10
    clip_eps = 0.2
    vf_coef = 0.5
    max_grad_norm = 0.5
    gamma = 0.99
    gae_lambda = 0.95

    # Build revisit mask array if previous cells are provided
    total_grid_size = int(np.prod(env.number_per_dim))
    if previous_cells is not None and len(previous_cells) > 0:
        revisit_mask_np = np.zeros(total_grid_size, dtype=bool)
        for cell in previous_cells:
            if isinstance(cell, (int, np.integer)):
                revisit_mask_np[int(cell)] = True
            else:
                flat = np.sum(np.asarray(cell, dtype=int) * env.strides)
                revisit_mask_np[flat] = True
        revisit_mask = jnp.asarray(revisit_mask_np)
    else:
        revisit_mask = None

    # Initialize model network and optimizer
    action_dim = len(env.u_min)
    obs_dim = env.model.n
    network = ActorCritic(action_dim=action_dim, pi_arch=pi_arch, vf_arch=vf_arch)

    rng, rng_init = jax.random.split(rng)
    init_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    params = network.init(rng_init, init_obs)

    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    rms_obs = init_running_mean_std((obs_dim,))

    # Initialize parallel environments
    rng, rng_envs = jax.random.split(rng)
    env_keys = jax.random.split(rng_envs, n_envs)
    init_states = jax.vmap(lambda k: _sample_safe_state(k, env))(env_keys)
    init_dists = jax.vmap(lambda s: jnp.linalg.norm(s - env.goal_center_jnp))(init_states)
    env_states = EnvState(
        state=init_states,
        steps=jnp.zeros((n_envs,), dtype=jnp.int32),
        prev_dist=init_dists,
    )
    obs = init_states

    # Step function for rollout collection across all vectorized environments
    def _step_fn(carry, _):
        t_state, r_obs, e_states, c_obs, k = carry
        k, k_act, k_step = jax.random.split(k, 3)

        n_obs = normalize_obs(r_obs, c_obs)
        actor_mean, log_std, val = t_state.apply_fn(t_state.params, n_obs)
        act = gaussian_sample(k_act, actor_mean, log_std)
        lp = gaussian_log_prob(act, actor_mean, log_std)

        step_keys = jax.random.split(k_step, n_envs)
        next_obs, next_env_states, rew, done, _ = jax.vmap(
            lambda rk, s, a: _env_step_jnp(rk, s, a, env, revisit_mask=revisit_mask)
        )(step_keys, e_states, act)

        trans = Transition(
            obs=n_obs,
            raw_obs=c_obs,
            action=act,
            value=val,
            reward=rew,
            done=done,
            log_prob=lp,
        )
        return (t_state, r_obs, next_env_states, next_obs, k), trans

    # Backward scan for Generalized Advantage Estimation (GAE)
    def _compute_gae(traj_batch, last_val):
        def _gae_step(carry, transition):
            gae, next_val = carry
            done = transition.done
            reward = transition.reward
            val = transition.value

            delta = reward + gamma * next_val * (1.0 - done) - val
            gae = delta + gamma * gae_lambda * (1.0 - done) * gae
            return (gae, val), (gae, gae + val)

        _, (advantages, targets) = jax.lax.scan(
            _gae_step,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
        )
        return advantages, targets

    # PPO minibatch loss and gradient update
    def _update_epoch(carry, _):
        t_state, traj_flat, k = carry
        k, k_perm = jax.random.split(k)
        perm = jax.random.permutation(k_perm, batch_size)
        shuffled_traj = jax.tree_util.tree_map(lambda x: x[perm], traj_flat)

        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (num_minibatches, minibatch_size) + x.shape[1:]),
            shuffled_traj,
        )

        def _update_minibatch(t_state, mb):
            def _loss_fn(p):
                actor_mean, log_std, val = t_state.apply_fn(p, mb.obs)
                lp = gaussian_log_prob(mb.action, actor_mean, log_std)
                entropy = gaussian_entropy(log_std)

                # Actor loss
                ratio = jnp.exp(lp - mb.log_prob)
                norm_adv = (mb.advantage - jnp.mean(mb.advantage)) / (jnp.std(mb.advantage) + 1e-8)
                actor_loss1 = -norm_adv * ratio
                actor_loss2 = -norm_adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                actor_loss = jnp.mean(jnp.maximum(actor_loss1, actor_loss2))

                # Critic loss with clipping
                v_clipped = mb.value + jnp.clip(val - mb.value, -clip_eps, clip_eps)
                v_loss1 = jnp.square(val - mb.target)
                v_loss2 = jnp.square(v_clipped - mb.target)
                critic_loss = 0.5 * jnp.mean(jnp.maximum(v_loss1, v_loss2))

                # Entropy
                mean_entropy = jnp.mean(entropy)
                ent_loss = -ent_coef * mean_entropy

                loss = actor_loss + vf_coef * critic_loss + ent_loss
                return loss, (actor_loss, critic_loss, mean_entropy)

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (_, (al, cl, ent)), grads = grad_fn(t_state.params)
            t_state = t_state.apply_gradients(grads=grads)
            return t_state, (al, cl, ent)

        t_state, (mb_als, mb_cls, mb_ents) = jax.lax.scan(_update_minibatch, t_state, minibatches)
        return (t_state, traj_flat, k), (jnp.mean(mb_als), jnp.mean(mb_cls), jnp.mean(mb_ents))

    # Single PPO update step
    @jax.jit
    def _update_step(runner_state, _):
        t_state, r_obs, e_states, c_obs, k = runner_state

        # 1. Rollout collection
        (t_state, r_obs, next_e_states, next_c_obs, k), traj_batch = jax.lax.scan(
            _step_fn,
            (t_state, r_obs, e_states, c_obs, k),
            None,
            length=n_steps,
        )

        mean_reward = jnp.mean(traj_batch.reward)

        # 2. Compute GAE
        norm_next_obs = normalize_obs(r_obs, next_c_obs)
        _, _, last_val = t_state.apply_fn(t_state.params, norm_next_obs)
        advantages, targets = _compute_gae(traj_batch, last_val)

        # 3. Update observation normalization statistics
        raw_obs_all = jnp.reshape(traj_batch.raw_obs, (batch_size, -1))
        r_obs = update_running_mean_std(r_obs, raw_obs_all)

        # 4. Flatten transitions
        traj_flat = FlatTransition(
            obs=jnp.reshape(traj_batch.obs, (batch_size, -1)),
            action=jnp.reshape(traj_batch.action, (batch_size, -1)),
            value=jnp.reshape(traj_batch.value, (batch_size,)),
            log_prob=jnp.reshape(traj_batch.log_prob, (batch_size,)),
            advantage=jnp.reshape(advantages, (batch_size,)),
            target=jnp.reshape(targets, (batch_size,)),
        )

        # 5. PPO optimization epochs
        (t_state, _, k), (ep_als, ep_cls, ep_ents) = jax.lax.scan(
            _update_epoch,
            (t_state, traj_flat, k),
            None,
            length=update_epochs,
        )

        update_metrics = {
            "mean_reward": mean_reward,
            "actor_loss": jnp.mean(ep_als),
            "critic_loss": jnp.mean(ep_cls),
            "entropy": jnp.mean(ep_ents),
        }

        next_runner_state = (t_state, r_obs, next_e_states, next_c_obs, k)
        return next_runner_state, update_metrics

    # Execute training loop in chunks for logging
    runner_state = (train_state, rms_obs, env_states, obs, rng)
    num_updates = max(1, total_timesteps // batch_size)
    chunk_updates = max(1, min(10, max(1, num_updates // 20)))
    num_chunks = max(1, int(np.ceil(num_updates / chunk_updates)))
    total_trained_steps = num_chunks * chunk_updates * batch_size

    @jax.jit
    def _run_chunk(state):
        return jax.lax.scan(_update_step, state, None, length=chunk_updates)

    t_start = time()
    pbar = tqdm(total=total_trained_steps, desc="PPO Training (PureJaxRL)", unit="step")
    all_metrics = []

    for _ in range(num_chunks):
        runner_state, chunk_metrics = _run_chunk(runner_state)
        chunk_metrics_np = {k: np.asarray(v) for k, v in chunk_metrics.items()}
        all_metrics.append(chunk_metrics_np)
        mean_rew = float(np.mean(chunk_metrics_np["mean_reward"]))
        pbar.set_postfix({"rew": f"{mean_rew:.2f}"})
        pbar.update(chunk_updates * batch_size)

    pbar.close()
    train_state, rms_obs, _, _, _ = runner_state
    jax.block_until_ready(train_state.params)
    logger.info(f"PPO training finished in {time() - t_start:.2f}s ({total_trained_steps} timesteps).")

    # Aggregate metric history
    metrics_history = {}
    if all_metrics:
        for k in all_metrics[0].keys():
            metrics_history[k] = np.concatenate([m[k].ravel() for m in all_metrics])

    # Plot learning curves
    output_dir = Path(getattr(args, "output_dir", "output"))
    if metrics_history and len(metrics_history.get("mean_reward", [])) > 1:
        plot_learning_curves(metrics_history, output_dir, total_timesteps=total_trained_steps)
        logger.info(f"- Learning curves plotted completed in {output_dir / 'rl_learning_curves.png'}")

    return network, train_state.params, rms_obs


# Plot training loss, reward, and entropy curves over PPO update iterations
def plot_learning_curves(metrics_history: Dict[str, np.ndarray], output_dir: Path, total_timesteps: int):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("PPO Learning Curves (PureJaxRL)", fontsize=13, fontweight="bold")

    n_pts = len(metrics_history["mean_reward"])
    timesteps = np.linspace(0, total_timesteps, n_pts)

    # 1. Mean Reward per step / batch
    ax = axs[0, 0]
    ax.plot(timesteps, metrics_history["mean_reward"], color="#2ca02c", linewidth=1.5)
    ax.set_title("Mean Step Reward", fontsize=11)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    # 2. Value Function Loss
    ax = axs[0, 1]
    ax.plot(timesteps, metrics_history["critic_loss"], color="#1f77b4", linewidth=1.5)
    ax.set_title("Critic (Value) Loss", fontsize=11)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)

    # 3. Policy Loss
    ax = axs[1, 0]
    ax.plot(timesteps, metrics_history["actor_loss"], color="#d62728", linewidth=1.5)
    ax.set_title("Actor (Policy) Loss", fontsize=11)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("PPO Clip Loss")
    ax.grid(True, alpha=0.3)

    # 4. Policy Entropy
    ax = axs[1, 1]
    ax.plot(timesteps, metrics_history["entropy"], color="#9467bd", linewidth=1.5)
    ax.set_title("Policy Entropy", fontsize=11)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Entropy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "rl_learning_curves.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(output_dir / "rl_learning_curves.png", format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# JIT-compiled policy mean prediction helper
@partial(jax.jit, static_argnums=(0,))
def _predict_policy_mean(apply_fn, params, rms_obs, obs_batch):
    norm_obs = normalize_obs(rms_obs, obs_batch)
    actor_mean, _, _ = apply_fn(params, norm_obs)
    return actor_mean


# Query policy for continuous actions and match to top-k nearest discrete grid actions
def find_policy_actions_batch(obs_batch, actor_critic, params, rms_obs, discrete_actions, num):
    obs_batch_jnp = jnp.asarray(obs_batch, dtype=jnp.float32)
    actions = np.asarray(_predict_policy_mean(actor_critic.apply, params, rms_obs, obs_batch_jnp))

    if num >= discrete_actions.shape[0]:
        top_k_idx = np.tile(np.arange(discrete_actions.shape[0]), (obs_batch.shape[0], 1))
    else:
        diff = actions[:, None, :] - discrete_actions[None, :, :]
        top_k_idx = np.argpartition(np.sum(diff * diff, axis=2), num - 1, axis=1)[:, :num]
    return discrete_actions[top_k_idx], discrete_actions[top_k_idx[:, 0]]


# Render 2D projections of trajectories, obstacle regions, and goal regions
def plot_rl_trajectories(base_model, eval_env, trajectories, dims, output_dir, max_trajectories=100):
    if len(dims) != 2:
        raise ValueError("This runner currently supports plotting exactly 2 dimensions.")

    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(8, 8))
    legend_handles = []

    def _add_boxes(boxes, color, alpha, label):
        if boxes is None or boxes.size == 0:
            return None
        rects = [
            mpatches.Rectangle(
                (box[0, dims[0]], box[0, dims[1]]),
                box[1, dims[0]] - box[0, dims[0]],
                box[1, dims[1]] - box[0, dims[1]],
            )
            for box in boxes
        ]
        ax.add_collection(PatchCollection(rects, facecolor=color, edgecolor="none", alpha=alpha, rasterized=True))
        return mpatches.Patch(color=color, alpha=alpha, label=label)

    if eval_env.critical.size > 0:
        h = _add_boxes(eval_env.critical, "red", 0.15, "Critical")
        if h:
            legend_handles.append(h)
    if eval_env.goal.size > 0:
        h = _add_boxes(eval_env.goal, "green", 0.25, "Goal")
        if h:
            legend_handles.append(h)
    if hasattr(eval_env.model, "charging_station") and eval_env.model.charging_station.size > 0:
        h = _add_boxes(eval_env.model.charging_station, "blue", 0.25, "Charging station")
        if h:
            legend_handles.append(h)

    if trajectories:
        selected_traces = [trace[:, dims] for trace in trajectories[:max_trajectories] if len(trace) > 0]
        if selected_traces:
            n_total = sum(len(t) for t in selected_traces) + len(selected_traces)
            combined = np.full((n_total, 2), np.nan, dtype=np.float32)
            offset = 0
            for trace in selected_traces:
                n = len(trace)
                combined[offset : offset + n] = trace
                offset += n + 1

            ax.plot(
                combined[:, 0],
                combined[:, 1],
                linewidth=1.0,
                alpha=0.9,
                color="black",
                marker=".",
                markersize=3.0,
                markeredgecolor="red",
                markerfacecolor="red",
                rasterized=True,
            )

    ax.set_xlim(eval_env.obs_low[dims[0]], eval_env.obs_high[dims[0]])
    ax.set_ylim(eval_env.obs_low[dims[1]], eval_env.obs_high[dims[1]])
    ax.set_xlabel(base_model.state_variables[dims[0]])
    ax.set_ylabel(base_model.state_variables[dims[1]])
    ax.set_title(f"PPO trajectories ({base_model.__class__.__name__})")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "rl_trajectories.pdf", format="pdf", bbox_inches="tight", dpi=200)
    plt.savefig(output_dir / "rl_trajectories.png", format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)


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
    actor_critic,
    params,
    rms_obs,
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


# D-dimensional summed area table (prefix sum) with periodic wrapping for O(2^D) box counting
class SpatialPrefixSum:
    def __init__(self, active_mask, number_per_dim, wrap):
        self.nd = np.asarray(number_per_dim, dtype=np.int64)
        self.D = len(self.nd)

        grid = active_mask.reshape(tuple(self.nd)).astype(np.int32)
        for d in range(self.D):
            if wrap[d]:
                grid = np.concatenate([grid, grid], axis=d)

        padded_shape = tuple(s + 1 for s in grid.shape)
        P = np.zeros(padded_shape, dtype=np.int32)
        P[tuple(slice(1, None) for _ in range(self.D))] = grid

        for d in range(self.D):
            np.cumsum(P, axis=d, out=P)

        padded_strides = np.ones(self.D, dtype=np.int64)
        for d in range(self.D - 2, -1, -1):
            padded_strides[d] = padded_strides[d + 1] * padded_shape[d + 1]

        self.prefix_flat = P.ravel()
        self.prefix_strides = padded_strides
        self.wrap = np.asarray(wrap, dtype=bool)

        corners = np.array(list(itertools.product([False, True], repeat=self.D)))
        self.corners = corners
        self.signs = np.where(np.sum(corners, axis=1) % 2 == 0, 1, -1).astype(np.int64)

    def count_boxes(self, raw_lbs, raw_ubs):
        batch_shape = raw_lbs.shape[:-1]
        query_lbs = np.empty_like(raw_lbs, dtype=np.int64)
        query_ubs = np.empty_like(raw_ubs, dtype=np.int64)
        empty = np.zeros(batch_shape, dtype=bool)

        for d in range(self.D):
            if self.wrap[d]:
                raw_span = raw_ubs[..., d] - raw_lbs[..., d] + 1
                span = np.clip(raw_span, 0, self.nd[d])
                query_lbs[..., d] = raw_lbs[..., d] % self.nd[d]
                query_ubs[..., d] = query_lbs[..., d] + span - 1
                empty |= (span <= 0)
            else:
                empty |= (raw_ubs[..., d] < 0) | (raw_lbs[..., d] >= self.nd[d])
                query_lbs[..., d] = np.clip(raw_lbs[..., d], 0, self.nd[d] - 1)
                query_ubs[..., d] = np.clip(raw_ubs[..., d], 0, self.nd[d] - 1)

        lb_contrib = query_lbs * self.prefix_strides
        ub_contrib = (query_ubs + 1) * self.prefix_strides

        counts = np.zeros(batch_shape, dtype=np.int64)
        for i in range(len(self.corners)):
            contrib = np.where(self.corners[i], lb_contrib, ub_contrib)
            counts += self.signs[i] * self.prefix_flat[np.sum(contrib, axis=-1)].astype(np.int64)

        counts[empty] = 0
        return counts


# Generate Cartesian product grid of integer coordinate offsets for inflation box
def _inflation_offsets(inflation_rate):
    axes = [np.arange(int(lo), int(hi) + 1) for lo, hi in inflation_rate]
    return np.stack([grid.ravel() for grid in np.meshgrid(*axes, indexing="ij")], axis=1)


# JAX jitted vectorized check for cell neighbor coordinates within grid bounds
@jax.jit
def _neighbors_in_bounds(cells, offsets, number_per_dim):
    def in_bounds_of(cell):
        neighbors = cell + offsets
        return jnp.all((neighbors >= 0) & (neighbors < number_per_dim), axis=1)

    return jax.vmap(in_bounds_of)(cells)


# Inflate set of visited cells into neighbor box and return deduplicated valid grid cells
def _inflate_cells(visited_cells, inflation_rate, number_per_dim):
    dim = len(number_per_dim)
    cells = np.asarray(list(visited_cells), dtype=np.int64).reshape(-1, dim)
    if len(cells) == 0:
        return np.zeros((0, dim), dtype=int)

    offsets = _inflation_offsets(inflation_rate)
    strides = _compute_linear_strides(number_per_dim)

    base_ids = cells @ strides
    offset_ids = offsets @ strides

    offsets_jax = jnp.asarray(offsets, dtype=jnp.int32)
    number_per_dim_jax = jnp.asarray(number_per_dim, dtype=jnp.int32)
    batch = max(1, INFLATE_IDS_PER_BATCH // len(offsets))

    batches = []
    for start in tqdm(range(0, len(cells), batch), desc="Inflate visited cells"):
        in_bounds = _neighbors_in_bounds(
            jnp.asarray(cells[start : start + batch], dtype=jnp.int32),
            offsets_jax,
            number_per_dim_jax,
        )
        ids = base_ids[start : start + batch, None] + offset_ids[None, :]
        batches.append(np.unique(np.where(np.asarray(in_bounds), ids, -1)))

    ids = np.unique(np.concatenate(batches)) if batches else np.zeros(0, dtype=np.int64)
    ids = ids[ids >= 0]
    return ((ids[:, None] // strides) % number_per_dim).astype(int)


# JAX jitted batched forward reachable set (FRS) interval bounds evaluation
@partial(jax.jit, static_argnums=(0,))
def _compute_batch_frs_bounds(step_set_fn, s_mins, s_maxs, actions_batch):
    def _per_state(s_min, s_max, actions):
        return jax.vmap(lambda u: step_set_fn(s_min, s_max, u, u))(actions)

    return jax.vmap(_per_state)(s_mins, s_maxs, actions_batch)


# Convert cell coordinates to continuous FRS interval boxes accounting for noise support
def _compute_frs_index_boxes(model, val_env, cell_coords, actions_batch, noise_support):
    s_mins = val_env.obs_low + cell_coords.astype(np.float32) * val_env.bin_widths
    s_maxs = s_mins + val_env.bin_widths
    frs_mins, frs_maxs = _compute_batch_frs_bounds(
        model.step_set, s_mins, s_maxs, jnp.asarray(actions_batch, dtype=jnp.float32)
    )
    raw_idx_lbs = np.floor((np.asarray(frs_mins, dtype=np.float32) - noise_support - val_env.obs_low) / val_env.bin_widths).astype(int)
    raw_idx_ubs = np.floor((np.asarray(frs_maxs, dtype=np.float32) + noise_support - val_env.obs_low) / val_env.bin_widths).astype(int)
    return raw_idx_lbs, raw_idx_ubs


# Vectorized extraction of discrete grid cells covered by reachability bounding boxes
def _extract_frs_cells_vectorized(raw_idx_lbs, raw_idx_ubs, number_per_dim, strides, wrap):
    N, K, D = raw_idx_lbs.shape
    spans = np.maximum(1, raw_idx_ubs - raw_idx_lbs + 1)
    max_spans = np.max(spans, axis=(0, 1))

    offset_axes = [np.arange(s, dtype=np.int64) for s in max_spans]
    offsets = offset_axes[0][:, None] if D == 1 else np.stack([g.ravel() for g in np.meshgrid(*offset_axes, indexing="ij")], axis=-1)

    M = offsets.shape[0]
    flat_cells = np.zeros((N, K, M), dtype=np.int64)
    valid_mask = np.ones((N, K, M), dtype=bool)

    for d in range(D):
        coord_d = raw_idx_lbs[:, :, d, None] + offsets[:, d][None, None, :]
        ub_d = raw_idx_ubs[:, :, d, None]
        num_d = number_per_dim[d]
        stride_d = strides[d]

        if wrap[d]:
            valid_mask &= (coord_d <= ub_d)
            flat_cells += (coord_d % num_d) * stride_d
        else:
            valid_mask &= (coord_d <= ub_d) & (coord_d >= 0) & (coord_d < num_d)
            flat_cells += np.clip(coord_d, 0, num_d - 1) * stride_d

    return flat_cells, valid_mask


# Forward reachable set (FRS) guided BFS expansion using prefix sum box queries
def _smart_inflate_cells(
    visited,
    model,
    val_env: JaxBenchmarkEnv,
    actor_critic: ActorCritic,
    params,
    rms_obs: RunningMeanStd,
    discrete_actions,
    args,
    number_per_dim,
    noise_support_ratio=0.5,
):
    dim = len(number_per_dim)
    visited_arr = np.asarray(list(visited), dtype=np.int64).reshape(-1, dim) if len(visited) > 0 else np.zeros((0, dim), dtype=np.int64)
    strides = _compute_linear_strides(number_per_dim)
    total_grid_size = int(np.prod(number_per_dim))

    active_states_mask = np.zeros(total_grid_size, dtype=bool)
    if len(visited_arr) > 0:
        active_states_mask[np.dot(visited_arr, strides)] = True

    wrap = np.asarray(getattr(model, "wrap", np.zeros(model.n, dtype=bool)), dtype=bool)

    if len(visited_arr) > 0:
        obs_batch_init = np.asarray(val_env.obs_low + (visited_arr.astype(np.float32) + 0.5) * val_env.bin_widths, dtype=np.float32)
        _, init_rl_actions = find_policy_actions_batch(obs_batch_init, actor_critic, params, rms_obs, discrete_actions, num=1)
    else:
        init_rl_actions = np.zeros((0, discrete_actions.shape[1]), dtype=np.float32)

    noise_support = 0.0
    if hasattr(model, "noise") and isinstance(model.noise, dict) and "support_radius" in model.noise:
        noise_support = model.noise["support_radius"] * noise_support_ratio

    # Phase 1: Expand reachable cells from initial rollout trajectory under greedy RL actions
    phase1_added_count = 0
    pbar_p1 = tqdm(total=len(visited_arr), desc="Phase 1: Batched FRS expansion")

    visited_arr_all = np.asarray(visited_arr, dtype=np.float32)
    init_rl_actions_all_batch = init_rl_actions[:, None, :]
    p1_new_queue_flats = []

    for ch_start in range(0, len(visited_arr), CHUNK_SIZE):
        ch_end = min(ch_start + CHUNK_SIZE, len(visited_arr))
        visited_chunk = visited_arr_all[ch_start:ch_end]
        init_actions_chunk = init_rl_actions_all_batch[ch_start:ch_end]

        raw_idx_lbs_p1, raw_idx_ubs_p1 = _compute_frs_index_boxes(model, val_env, visited_chunk, init_actions_chunk, noise_support)
        flat_cells_p1, valid_mask_p1 = _extract_frs_cells_vectorized(raw_idx_lbs_p1, raw_idx_ubs_p1, number_per_dim, strides, wrap)

        flats_p1 = flat_cells_p1[:, 0, :]
        valid_p1 = valid_mask_p1[:, 0, :]

        new_flats_p1 = flats_p1[valid_p1 & (~active_states_mask[flats_p1])]
        if len(new_flats_p1) > 0:
            unique_flats = np.unique(new_flats_p1)
            active_states_mask[unique_flats] = True
            p1_new_queue_flats.append(unique_flats)
            phase1_added_count += len(unique_flats)

        pbar_p1.update(ch_end - ch_start)

    pbar_p1.close()

    queue_flats = np.unique(np.concatenate(p1_new_queue_flats)) if p1_new_queue_flats else np.array([], dtype=np.int64)
    logger.info(f"- Phase 1 complete: Added {phase1_added_count} states. Queue size: {len(queue_flats)}.")

    # Phase 2: Breadth-first search queue expansion with prefix-sum action scoring
    prefix_sum = SpatialPrefixSum(active_states_mask, number_per_dim, wrap)
    phase2_added_count = 0
    p2_iter = 0

    while len(queue_flats) > 0:
        p2_iter += 1
        N_q = len(queue_flats)
        pbar_p2 = tqdm(total=N_q, desc=f"Phase 2 (iter {p2_iter}): Queue FRS expansion", unit="state")

        queue_coords = np.stack(np.unravel_index(queue_flats, number_per_dim), axis=-1)
        obs_batch_queue = np.asarray(val_env.obs_low + (queue_coords.astype(np.float32) + 0.5) * val_env.bin_widths, dtype=np.float32)
        queue_actions_all, _ = find_policy_actions_batch(
            obs_batch_queue, actor_critic, params, rms_obs, discrete_actions, num=args.RL_actions_per_state
        )

        p2_iter_new_flats = []
        for ch_start in range(0, N_q, CHUNK_SIZE):
            ch_end = min(ch_start + CHUNK_SIZE, N_q)
            queue_chunk = queue_coords[ch_start:ch_end]
            actions_chunk = queue_actions_all[ch_start:ch_end]

            raw_idx_lbs_q, raw_idx_ubs_q = _compute_frs_index_boxes(model, val_env, queue_chunk, actions_chunk, noise_support)
            active_counts = prefix_sum.count_boxes(raw_idx_lbs_q, raw_idx_ubs_q)
            best_act_indices = np.argmax(active_counts, axis=1)

            N_chunk = ch_end - ch_start
            win_lbs = raw_idx_lbs_q[np.arange(N_chunk), best_act_indices][:, None, :]
            win_ubs = raw_idx_ubs_q[np.arange(N_chunk), best_act_indices][:, None, :]

            flat_cells_q, valid_mask_q = _extract_frs_cells_vectorized(win_lbs, win_ubs, number_per_dim, strides, wrap)
            win_flats = flat_cells_q[:, 0, :]
            win_valid = valid_mask_q[:, 0, :]

            chunk_new_flats = win_flats[win_valid & (~active_states_mask[win_flats])]
            if len(chunk_new_flats) > 0:
                p2_iter_new_flats.append(chunk_new_flats)

            pbar_p2.update(N_chunk)

        if p2_iter_new_flats:
            unique_new = np.unique(np.concatenate(p2_iter_new_flats))
            active_states_mask[unique_new] = True
            queue_flats = unique_new
            added_this_iter = len(unique_new)
            phase2_added_count += added_this_iter
            prefix_sum = SpatialPrefixSum(active_states_mask, number_per_dim, wrap)
        else:
            queue_flats = np.array([], dtype=np.int64)
            added_this_iter = 0

        pbar_p2.set_postfix({"added_iter": added_this_iter, "total_active": int(np.sum(active_states_mask))})
        pbar_p2.close()

    logger.info(f"- Phase 2 complete: Added {phase2_added_count} states. Total active states: {int(np.sum(active_states_mask))}.")
    all_active_flats = np.where(active_states_mask)[0]
    return np.stack(np.unravel_index(all_active_flats, number_per_dim), axis=-1).astype(int)


# Main pipeline: PureJaxRL PPO training, policy evaluation, tube expansion, and action extraction
def find_active(model, args, previous_cells=None):
    cfg = RLConfig(
        max_steps=args.max_steps,
        goal_reward=args.goal_reward,
        unsafe_penalty=args.unsafe_penalty,
        out_of_bounds_penalty=args.out_of_bounds_penalty,
        revisit_penalty=args.revisit_penalty,
        distance_reward=getattr(args, "distance_reward", 0.0),
        per_step_reward=getattr(args, "per_step_reward", 0.0),
    )

    env = JaxBenchmarkEnv(model, cfg, previous_cells=previous_cells)

    pi_arch = tuple(args.pi_arch if args.pi_arch is not None else model.pi_arch)
    vf_arch = tuple(args.vf_arch if args.vf_arch is not None else model.vf_arch)

    actor_critic, params, rms_obs = train_ppo(
        env=env,
        args=args,
        pi_arch=pi_arch,
        vf_arch=vf_arch,
        previous_cells=previous_cells,
        seed=args.seed,
    )

    # Discretize continuous control space into Cartesian grid of discrete actions
    discrete_actions_per_dim = [
        np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i])
        for i in range(len(model.num_actions))
    ]
    discrete_actions = np.array(list(itertools.product(*discrete_actions_per_dim)), dtype=np.float32)

    # Run evaluation rollouts to identify visited state cells
    goal_reached, newly_visited, total_cells = evaluate_policy(
        actor_critic=actor_critic,
        params=params,
        rms_obs=rms_obs,
        base_model=model,
        env=env,
        cfg=cfg,
        episodes=args.eval_episodes,
        dims=list(model.plot_dimensions),
        args=args,
        discrete_actions=discrete_actions,
        seed=args.seed,
    )

    logger.info(f"Goal reached in {goal_reached}/{args.eval_episodes} episodes.")
    t = time()

    # Expand visited cell tube using standard box inflation or smart FRS reachable expansion
    number_per_dim = np.asarray(model.partition["number_per_dim"], dtype=np.int64)

    if args.tube_method == "inflation":
        active_states = _inflate_cells(newly_visited, model.inflation_rate, number_per_dim)
    elif args.tube_method == "smart":
        active_states = _smart_inflate_cells(
            visited=newly_visited,
            model=model,
            val_env=env,
            actor_critic=actor_critic,
            params=params,
            rms_obs=rms_obs,
            discrete_actions=discrete_actions,
            args=args,
            number_per_dim=number_per_dim,
        )
    else:
        raise ValueError(f"Unknown tube_method: {args.tube_method}")

    logger.info(f"- Active states extraction completed in {time() - t:.2f} seconds.")
    t = time()

    # Extract top-K discrete actions and greedy policy for all active states
    obs_batch = np.asarray(env.obs_low + (active_states.astype(np.float32) + 0.5) * env.bin_widths, dtype=np.float32)
    top_k, rl_policy = find_policy_actions_batch(
        obs_batch, actor_critic, params, rms_obs, discrete_actions, num=args.RL_actions_per_state
    )
    active_actions = {tuple(cell): top_k[i] for i, cell in enumerate(active_states.tolist())}

    logger.info(f"- Active state/action extraction completed in {time() - t:.2f} seconds.")
    return active_states, active_actions, rl_policy
