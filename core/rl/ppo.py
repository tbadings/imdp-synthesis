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
from .env import BenchmarkEnv, EnvState, _sample_safe_state, _env_step_jnp
from .policy import (
    ActorCritic,
    gaussian_sample,
    gaussian_log_prob,
    gaussian_entropy,
)

logger = logging.getLogger(__name__)


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    terminated: jnp.ndarray
    next_value: jnp.ndarray
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
    env: BenchmarkEnv,
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
    chunk_updates = max(1, min(10, max(1, num_updates // 20)))
    num_chunks = max(1, int(np.ceil(num_updates / chunk_updates)))
    total_updates = num_chunks * chunk_updates
    update_epochs = args.update_epochs
    clip_eps = args.clip_eps
    vf_coef = args.vf_coef
    max_grad_norm = args.max_grad_norm
    gamma = args.gamma
    gae_lambda = args.gae_lambda
    adam_eps = args.adam_eps

    # Initialize model network and optimizer
    action_dim = len(env.u_min)
    obs_dim = env.model.n
    network = ActorCritic(action_dim=action_dim, pi_arch=pi_arch, vf_arch=vf_arch)

    rng, rng_init = jax.random.split(rng)
    init_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    params = network.init(rng_init, init_obs)

    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate, eps=adam_eps),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # Initialize parallel environments
    rng, rng_envs = jax.random.split(rng)
    env_keys = jax.random.split(rng_envs, n_envs)
    init_states = jax.vmap(lambda k: _sample_safe_state(k, env))(env_keys)
    init_dists = jax.vmap(env.distance_to_goal)(init_states)
    env_states = EnvState(
        state=init_states,
        steps=jnp.zeros((n_envs,), dtype=jnp.int32),
        prev_dist=init_dists,
    )

    # Step function for rollout collection across all vectorized environments
    def _step_fn(carry, _):
        t_state, e_states, k = carry
        k, k_act, k_step = jax.random.split(k, 3)

        obs = e_states.state
        actor_mean, log_std, val = t_state.apply_fn(t_state.params, obs)
        act = gaussian_sample(k_act, actor_mean, log_std)
        lp = gaussian_log_prob(act, actor_mean, log_std)

        step_keys = jax.random.split(k_step, n_envs)
        _, next_env_states, rew, done, info = jax.vmap(
            lambda rk, s, a: _env_step_jnp(rk, s, a, env)
        )(step_keys, e_states, act)

        # Value of the successor *before* the auto-reset, so that episodes ending on the
        # step limit bootstrap from where they actually were instead of being treated as
        # terminal. Without this the critic charges every timeout the full remaining
        # step cost with no continuation value, which makes deliberately terminating
        # (crashing / leaving the domain) look cheaper than surviving.
        _, _, next_val = t_state.apply_fn(t_state.params, info["next_state"])

        trans = Transition(
            obs=obs,
            action=act,
            value=val,
            reward=rew,
            done=done,
            terminated=info["terminated"],
            next_value=next_val,
            log_prob=lp,
        )
        return (t_state, next_env_states, k), trans

    # Backward scan for Generalized Advantage Estimation (GAE)
    def _compute_gae(traj_batch, last_val):
        del last_val  # each transition carries the value of its own successor

        def _gae_step(gae, transition):
            done = transition.done.astype(jnp.float32)

            # Bootstrap on the true successor, zeroed only on *real* terminations.
            # Time-limit truncations keep their continuation value.
            next_val = jnp.where(transition.terminated, 0.0, transition.next_value)
            delta = transition.reward + gamma * next_val - transition.value
            gae = delta + gamma * gae_lambda * (1.0 - done) * gae
            return gae, (gae, gae + transition.value)

        _, (advantages, targets) = jax.lax.scan(
            _gae_step,
            jnp.zeros_like(traj_batch.value[0]),
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
        t_state, e_states, k = runner_state

        # 1. Rollout collection
        (t_state, next_e_states, k), traj_batch = jax.lax.scan(
            _step_fn,
            (t_state, e_states, k),
            None,
            length=n_steps,
        )

        # Diagnostics: how many episodes ended, and how many of those reached the goal.
        num_terminated = jnp.sum(traj_batch.terminated)
        num_goal = jnp.sum(traj_batch.terminated & (traj_batch.reward > 0))
        mean_reward = jnp.mean(traj_batch.reward)

        # 2. Compute GAE
        advantages, targets = _compute_gae(traj_batch, None)

        # 3. Flatten transitions
        traj_flat = FlatTransition(
            obs=jnp.reshape(traj_batch.obs, (batch_size, -1)),
            action=jnp.reshape(traj_batch.action, (batch_size, -1)),
            value=jnp.reshape(traj_batch.value, (batch_size,)),
            log_prob=jnp.reshape(traj_batch.log_prob, (batch_size,)),
            advantage=jnp.reshape(advantages, (batch_size,)),
            target=jnp.reshape(targets, (batch_size,)),
        )

        # 4. PPO optimization epochs
        (t_state, _, k), _ = jax.lax.scan(
            _update_epoch,
            (t_state, traj_flat, k),
            None,
            length=update_epochs,
        )

        next_runner_state = (t_state, next_e_states, k)
        return next_runner_state, (mean_reward, num_goal, num_terminated)

    # Training loop in chunks for logging
    runner_state = (train_state, env_states, rng)
    total_trained_steps = total_updates * batch_size

    @jax.jit
    def _run_chunk(state):
        return jax.lax.scan(_update_step, state, None, length=chunk_updates)

    t_start = time()
    pbar = tqdm(total=total_trained_steps, desc="PPO Training (PureJaxRL)", unit="step")

    for _ in range(num_chunks):
        runner_state, (chunk_rewards, chunk_goals, chunk_terms) = _run_chunk(runner_state)
        goals, terms = float(np.sum(chunk_goals)), float(np.sum(chunk_terms))
        pbar.set_postfix({
            "mean reward": f"{float(np.mean(chunk_rewards)):.2f}",
            "goal rate": f"{goals / max(terms, 1.0):.2f}",
        })
        pbar.update(chunk_updates * batch_size)

    pbar.close()
    train_state, _, _ = runner_state
    jax.block_until_ready(train_state.params)
    logger.info(f"PPO training finished in {time() - t_start:.2f}s ({total_trained_steps} timesteps).")

    return network, train_state.params
