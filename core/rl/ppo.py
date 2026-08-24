import logging
from pathlib import Path
from time import time
from typing import NamedTuple, Sequence

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from .config import RLConfig
from .env import JaxBenchmarkEnv, EnvState, _sample_safe_state, _env_step_jnp
from .models import (
    ActorCritic,
    RunningMeanStd,
    init_running_mean_std,
    update_running_mean_std,
    normalize_obs,
    gaussian_sample,
    gaussian_log_prob,
    gaussian_entropy,
)

logger = logging.getLogger(__name__)


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
            lambda rk, s, a: _env_step_jnp(rk, s, a, env)
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

                return actor_loss + vf_coef * critic_loss + ent_loss

            grads = jax.grad(_loss_fn)(t_state.params)
            t_state = t_state.apply_gradients(grads=grads)
            return t_state, None

        t_state, _ = jax.lax.scan(_update_minibatch, t_state, minibatches)
        return (t_state, traj_flat, k), None

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
        (t_state, _, k), _ = jax.lax.scan(
            _update_epoch,
            (t_state, traj_flat, k),
            None,
            length=update_epochs,
        )

        next_runner_state = (t_state, r_obs, next_e_states, next_c_obs, k)
        return next_runner_state, mean_reward

    # Training loop in chunks for logging
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

    for _ in range(num_chunks):
        runner_state, chunk_rewards = _run_chunk(runner_state)
        mean_rew = float(np.mean(chunk_rewards))
        pbar.set_postfix({"rew": f"{mean_rew:.2f}"})
        pbar.update(chunk_updates * batch_size)

    pbar.close()
    train_state, rms_obs, _, _, _ = runner_state
    jax.block_until_ready(train_state.params)
    logger.info(f"PPO training finished in {time() - t_start:.2f}s ({total_trained_steps} timesteps).")

    return network, train_state.params, rms_obs
