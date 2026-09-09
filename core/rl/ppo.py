from typing import NamedTuple, Sequence
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .base import BaseRL


class ActorCritic(nn.Module):
    action_dim: int
    pi_arch: Sequence[int] = (64, 64)
    vf_arch: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, x):
        actor_x, critic_x = x, x
        for h in self.pi_arch:
            actor_x = nn.relu(nn.Dense(h, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.zeros)(actor_x))
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01), bias_init=nn.initializers.zeros)(actor_x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))

        for h in self.vf_arch:
            critic_x = nn.relu(nn.Dense(h, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.zeros)(critic_x))
        critic_val = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0), bias_init=nn.initializers.zeros)(critic_x)

        return actor_mean, log_std, jnp.squeeze(critic_val, axis=-1)


def gaussian_sample(rng, mean, log_std):
    return mean + jnp.exp(log_std) * jax.random.normal(rng, shape=mean.shape)


def gaussian_log_prob(action, mean, log_std):
    return -0.5 * jnp.sum(jnp.square((action - mean) / jnp.exp(log_std)) + 2.0 * log_std + jnp.log(2.0 * jnp.pi), axis=-1)


def gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)), axis=-1)


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray


def make_train(env, cfg):
    """Create JAX PPO training components."""
    lr = cfg.learning_rate
    n_envs = cfg.n_envs
    n_steps = cfg.n_steps
    update_epochs = cfg.update_epochs
    rl_batch_size = cfg.rl_batch_size
    gamma, gae_lambda = cfg.gamma, cfg.gae_lambda
    clip_eps = cfg.clip_eps
    vf_coef, ent_coef = cfg.vf_coef, cfg.ent_coef
    max_grad_norm, adam_eps = cfg.max_grad_norm, cfg.adam_eps

    total_batch = n_envs * n_steps
    num_minibatches = total_batch // rl_batch_size

    network = ActorCritic(
        action_dim=env.action_dim,
        pi_arch=tuple(cfg.pi_arch),
        vf_arch=tuple(cfg.vf_arch),
    )

    def init_train_state(rng: jax.Array) -> TrainState:
        dummy_obs = jnp.zeros((1, env.obs_dim))
        params = network.init(rng, dummy_obs)
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr, eps=adam_eps))
        return TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    def step_env(carry, _):
        t_state, env_states, rng = carry
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        mean, log_std, val = t_state.apply_fn(t_state.params, env_states.obs)
        action = gaussian_sample(rng_act, mean, log_std)
        log_prob = gaussian_log_prob(action, mean, log_std)

        step_keys = jax.random.split(rng_step, n_envs)
        next_obs, next_env_states, rew, done, _ = env.step(step_keys, env_states, action)

        transition = Transition(
            obs=env_states.obs, action=action, value=val, reward=rew, done=done, log_prob=log_prob
        )
        return (t_state, next_env_states, rng), transition

    def compute_gae(traj_batch, last_val):
        def _gae_step(gae, transition_and_next_val):
            trans, next_v = transition_and_next_val
            delta = trans.reward + gamma * next_v * (1.0 - trans.done) - trans.value
            gae = delta + gamma * gae_lambda * (1.0 - trans.done) * gae
            return gae, (gae, gae + trans.value)

        next_values = jnp.concatenate([traj_batch.value[1:], jnp.expand_dims(last_val, axis=0)], axis=0)
        _, (advantages, targets) = jax.lax.scan(
            _gae_step, jnp.zeros_like(traj_batch.value[0]), (traj_batch, next_values), reverse=True
        )
        return advantages, targets

    def update_epoch(carry, _):
        t_state, traj_flat, advs, targets, rng = carry
        rng, rng_perm = jax.random.split(rng)
        perm = jax.random.permutation(rng_perm, total_batch)

        shuffled_trans = jax.tree_util.tree_map(lambda x: x[perm], traj_flat)
        shuffled_advs = advs[perm]
        shuffled_targets = targets[perm]

        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (num_minibatches, rl_batch_size) + x.shape[1:]),
            (shuffled_trans, shuffled_advs, shuffled_targets),
        )

        def update_minibatch(t_state, mb_data):
            mb_trans, mb_adv, mb_target = mb_data

            def loss_fn(params):
                mean, log_std, val = t_state.apply_fn(params, mb_trans.obs)
                lp = gaussian_log_prob(mb_trans.action, mean, log_std)
                entropy = gaussian_entropy(log_std)

                log_ratio = jnp.clip(lp - mb_trans.log_prob, -20.0, 20.0)
                ratio = jnp.exp(log_ratio)
                actor_loss1 = -mb_adv * ratio
                actor_loss2 = -mb_adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                actor_loss = jnp.mean(jnp.maximum(actor_loss1, actor_loss2))
                critic_loss = 0.5 * jnp.mean(jnp.square(val - mb_target))
                total_loss = actor_loss + vf_coef * critic_loss - ent_coef * jnp.mean(entropy)
                approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
                return total_loss, (actor_loss, critic_loss, jnp.mean(entropy), approx_kl)

            grads, (act_l, crit_l, ent, kl) = jax.grad(loss_fn, has_aux=True)(t_state.params)
            t_state = t_state.apply_gradients(grads=grads)
            return t_state, {"actor_loss": act_l, "critic_loss": crit_l, "entropy": ent, "approx_kl": kl}

        t_state, mb_metrics = jax.lax.scan(update_minibatch, t_state, minibatches)
        return (t_state, traj_flat, advs, targets, rng), jax.tree_util.tree_map(jnp.mean, mb_metrics)

    def update_step(runner_state, _):
        t_state, env_states, rng = runner_state
        (t_state, next_env_states, rng), traj_batch = jax.lax.scan(step_env, (t_state, env_states, rng), None, length=n_steps)
        _, _, last_val = t_state.apply_fn(t_state.params, next_env_states.obs)

        advantages, targets = compute_gae(traj_batch, last_val)
        norm_adv = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        traj_flat = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (total_batch,) + x.shape[2:]), traj_batch)
        advs_flat = jnp.reshape(norm_adv, (total_batch,))
        targets_flat = jnp.reshape(targets, (total_batch,))

        (t_state, _, _, _, rng), epoch_metrics = jax.lax.scan(
            update_epoch, (t_state, traj_flat, advs_flat, targets_flat, rng), None, length=update_epochs
        )

        metrics = {
            "reward": jnp.mean(traj_batch.reward),
            "actor_loss": jnp.mean(epoch_metrics["actor_loss"]),
            "critic_loss": jnp.mean(epoch_metrics["critic_loss"]),
            "entropy": jnp.mean(epoch_metrics["entropy"]),
            "approx_kl": jnp.mean(epoch_metrics["approx_kl"]),
        }
        return (t_state, next_env_states, rng), metrics

    return network, init_train_state, update_step


class PPO(BaseRL):
    """PPO reinforcement learning algorithm in JAX."""

    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.network, self.init_train_state, self.update_step = make_train(env, cfg)

    def train(self, seed: int = 0):
        rng = jax.random.PRNGKey(seed)
        rng, rng_init, rng_envs = jax.random.split(rng, 3)

        train_state = self.init_train_state(rng_init)
        _, init_env_states = self.env.reset(jax.random.split(rng_envs, self.cfg.n_envs))
        runner_state = (train_state, init_env_states, rng)

        format_fn = lambda m, rew: {
            "rew": f"{rew:.2f}",
            "pi_loss": f"{m['actor_loss']:.3f}",
            "v_loss": f"{m['critic_loss']:.3f}",
            "ent": f"{m['entropy']:.2f}",
            "kl": f"{m['approx_kl']:.4f}",
        }
        runner_state = self._run_training(
            "Training PPO (JAX)",
            runner_state,
            self.update_step,
            steps_per_update=self.cfg.n_envs * self.cfg.n_steps,
            num_chunks=20,
            format_fn=format_fn,
        )

        self.params = runner_state[0].params
        return self

    def get_predict_fn(self):
        return lambda norm_obs: self.network.apply(self.params, norm_obs)[0]
