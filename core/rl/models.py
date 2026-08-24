from functools import partial
from typing import NamedTuple, Sequence
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


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
