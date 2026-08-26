from functools import partial
from typing import Sequence
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
def _predict_policy_mean(apply_fn, params, obs_batch):
    actor_mean, _, _ = apply_fn(params, obs_batch)
    return actor_mean


# Query policy for continuous actions and match to top-k nearest discrete grid actions
def find_policy_actions_batch(obs_batch, actor_critic, params, discrete_actions, num):
    obs_batch_jnp = jnp.asarray(obs_batch, dtype=jnp.float32)
    actions = np.asarray(_predict_policy_mean(actor_critic.apply, params, obs_batch_jnp))

    num = min(num, discrete_actions.shape[0])
    diff = actions[:, None, :] - discrete_actions[None, :, :]
    top_k_idx = np.argsort(np.sum(diff * diff, axis=2), axis=1)[:, :num]
    
    return discrete_actions[top_k_idx], discrete_actions[top_k_idx[:, 0]]
