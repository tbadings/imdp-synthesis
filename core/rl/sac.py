from typing import NamedTuple, Sequence
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from .base import BaseRL


class SACActor(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = nn.relu(nn.Dense(h)(x))
        mean = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(nn.Dense(self.action_dim)(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, mean, log_std, rng):
        std = jnp.exp(log_std)
        normal_sample = jax.random.normal(rng, shape=mean.shape)
        action = jnp.tanh(mean + std * normal_sample)
        log_prob = -0.5 * jnp.sum(jnp.square(normal_sample) + 2.0 * log_std + jnp.log(2.0 * jnp.pi), axis=-1)
        log_prob -= jnp.sum(jnp.log(1.0 - jnp.square(action) + 1e-6), axis=-1)
        return action, log_prob


class SACCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        q1 = x
        for h in self.hidden_dims:
            q1 = nn.relu(nn.Dense(h)(q1))
        q1 = jnp.squeeze(nn.Dense(1)(q1), axis=-1)

        q2 = x
        for h in self.hidden_dims:
            q2 = nn.relu(nn.Dense(h)(q2))
        q2 = jnp.squeeze(nn.Dense(1)(q2), axis=-1)
        return q1, q2


class ReplayBuffer(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray
    ptr: int
    size: int
    capacity: int

    @classmethod
    def create(cls, capacity: int, obs_dim: int, action_dim: int):
        return cls(
            obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
            action=jnp.zeros((capacity, action_dim), dtype=jnp.float32),
            reward=jnp.zeros((capacity,), dtype=jnp.float32),
            next_obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
            done=jnp.zeros((capacity,), dtype=jnp.float32),
            ptr=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
            capacity=capacity,
        )

    def add(self, obs, action, reward, next_obs, done):
        n = obs.shape[0]
        idx = (self.ptr + jnp.arange(n)) % self.capacity
        return self._replace(
            obs=self.obs.at[idx].set(obs),
            action=self.action.at[idx].set(action),
            reward=self.reward.at[idx].set(reward),
            next_obs=self.next_obs.at[idx].set(next_obs),
            done=self.done.at[idx].set(done),
            ptr=(self.ptr + n) % self.capacity,
            size=jnp.minimum(self.size + n, self.capacity),
        )

    def sample(self, rng: jax.Array, batch_size: int):
        idx = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=jnp.maximum(self.size, 1))
        return self.obs[idx], self.action[idx], self.reward[idx], self.next_obs[idx], self.done[idx]


class SACTrainState(NamedTuple):
    actor: TrainState
    critic: TrainState
    target_critic_params: dict
    log_alpha: TrainState


def make_train(env, cfg):
    """Create JAX SAC training components."""
    lr = cfg.learning_rate
    gamma, tau = cfg.gamma, cfg.tau
    batch_size = cfg.sac_batch_size
    n_envs = cfg.n_envs
    buffer_size = cfg.buffer_size
    warmup_steps = cfg.warmup_steps
    min_alpha = cfg.min_alpha
    min_log_alpha = float(jnp.log(min_alpha))
    target_entropy = -float(env.action_dim)

    actor_net = SACActor(action_dim=env.action_dim, hidden_dims=tuple(cfg.pi_arch))
    critic_net = SACCritic(hidden_dims=tuple(cfg.vf_arch))

    def init_train_state(rng: jax.Array):
        rng_act, rng_crit, _ = jax.random.split(rng, 3)
        dummy_obs = jnp.zeros((1, env.obs_dim))
        dummy_act = jnp.zeros((1, env.action_dim))

        actor_state = TrainState.create(apply_fn=actor_net.apply, params=actor_net.init(rng_act, dummy_obs), tx=optax.adam(lr))
        critic_params = critic_net.init(rng_crit, dummy_obs, dummy_act)
        critic_state = TrainState.create(apply_fn=critic_net.apply, params=critic_params, tx=optax.adam(lr))
        alpha_state = TrainState.create(
            apply_fn=lambda p, x: jnp.exp(p["log_alpha"]),
            params={"log_alpha": jnp.array(0.0, dtype=jnp.float32)},
            tx=optax.adam(lr),
        )

        buffer = ReplayBuffer.create(buffer_size, env.obs_dim, env.action_dim)
        target_critic_params = critic_net.init(rng_crit, dummy_obs, dummy_act)
        return SACTrainState(actor=actor_state, critic=critic_state, target_critic_params=target_critic_params, log_alpha=alpha_state), buffer

    def update_step(runner_state, _):
        t_state, env_states, buffer, rng = runner_state
        rng, rng_act, rng_rand, rng_step, rng_sample, rng_next_act, rng_actor = jax.random.split(rng, 7)

        mean, log_std = t_state.actor.apply_fn(t_state.actor.params, env_states.obs)
        policy_action, _ = actor_net.sample(mean, log_std, rng_act)
        random_action = jax.random.uniform(rng_rand, shape=policy_action.shape, minval=-1.0, maxval=1.0)
        action = jnp.where(buffer.size < warmup_steps, random_action, policy_action)

        step_keys = jax.random.split(rng_step, n_envs)
        next_obs, next_env_states, rew, done, info = env.step(step_keys, env_states, action)
        buffer = buffer.add(env_states.obs, action, rew, info["next_obs"], info["terminated"].astype(jnp.float32))

        b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(rng_sample, batch_size)
        alpha = jnp.maximum(jnp.exp(t_state.log_alpha.params["log_alpha"]), min_alpha)

        def _do_update(ts):
            next_mean, next_log_std = ts.actor.apply_fn(ts.actor.params, b_next_obs)
            next_act, next_lp = actor_net.sample(next_mean, next_log_std, rng_next_act)
            target_q1, target_q2 = critic_net.apply(ts.target_critic_params, b_next_obs, next_act)
            target_q = jnp.minimum(target_q1, target_q2) - alpha * next_lp
            y = jax.lax.stop_gradient(b_rew + gamma * (1.0 - b_done) * target_q)

            def critic_loss_fn(params):
                q1, q2 = ts.critic.apply_fn(params, b_obs, b_act)
                return 0.5 * jnp.mean(jnp.square(q1 - y)) + 0.5 * jnp.mean(jnp.square(q2 - y)), jnp.mean(jnp.minimum(q1, q2))

            (critic_loss, q_mean), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(ts.critic.params)
            new_critic = ts.critic.apply_gradients(grads=critic_grads)

            def actor_loss_fn(params):
                m, ls = ts.actor.apply_fn(params, b_obs)
                sampled_act, lp = actor_net.sample(m, ls, rng_actor)
                q1, q2 = new_critic.apply_fn(new_critic.params, b_obs, sampled_act)
                return jnp.mean(alpha * lp - jnp.minimum(q1, q2)), lp

            (actor_loss, b_lp), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(ts.actor.params)
            new_actor = ts.actor.apply_gradients(grads=actor_grads)

            def alpha_loss_fn(params):
                return -jnp.mean(jnp.exp(params["log_alpha"]) * (b_lp + target_entropy))

            alpha_grads = jax.grad(alpha_loss_fn)(ts.log_alpha.params)
            new_log_alpha = ts.log_alpha.apply_gradients(grads=alpha_grads)
            clamped = jnp.maximum(new_log_alpha.params["log_alpha"], min_log_alpha)
            new_log_alpha = new_log_alpha.replace(params={"log_alpha": clamped})

            new_target_params = jax.tree_util.tree_map(lambda n, o: tau * n + (1.0 - tau) * o, new_critic.params, ts.target_critic_params)
            updated_state = SACTrainState(actor=new_actor, critic=new_critic, target_critic_params=new_target_params, log_alpha=new_log_alpha)
            return updated_state, {"critic_loss": critic_loss, "actor_loss": actor_loss, "q": q_mean, "alpha": alpha, "entropy": -jnp.mean(b_lp)}

        dummy_metrics = {"critic_loss": jnp.array(0.0), "actor_loss": jnp.array(0.0), "q": jnp.array(0.0), "alpha": alpha, "entropy": jnp.array(0.0)}
        new_t_state, step_metrics = jax.lax.cond(buffer.size >= warmup_steps, _do_update, lambda ts: (ts, dummy_metrics), t_state)

        return (new_t_state, next_env_states, buffer, rng), {"reward": jnp.mean(rew), **step_metrics}

    return (actor_net, critic_net), init_train_state, update_step


class SAC(BaseRL):
    """SAC reinforcement learning algorithm in JAX."""

    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        (self.actor_net, self.critic_net), self.init_train_state, self.update_step = make_train(env, cfg)

    def train(self, seed: int = 0):
        rng = jax.random.PRNGKey(seed)
        rng, rng_init, rng_envs = jax.random.split(rng, 3)

        train_state, buffer = self.init_train_state(rng_init)
        _, init_env_states = self.env.reset(jax.random.split(rng_envs, self.cfg.n_envs))
        runner_state = (train_state, init_env_states, buffer, rng)

        format_fn = lambda m, rew: {
            "rew": f"{rew:.2f}",
            "q_loss": f"{m['critic_loss']:.3f}",
            "q": f"{m['q']:.2f}",
            "pi_loss": f"{m['actor_loss']:.3f}",
            "alpha": f"{m['alpha']:.3f}",
            "ent": f"{m['entropy']:.2f}",
        }
        runner_state = self._run_training(
            "Training SAC (JAX)",
            runner_state,
            self.update_step,
            steps_per_update=self.cfg.n_envs,
            num_chunks=25,
            format_fn=format_fn,
        )

        self.params = runner_state[0].actor.params
        return self

    def get_predict_fn(self):
        return lambda norm_obs: jnp.tanh(self.actor_net.apply(self.params, norm_obs)[0])
