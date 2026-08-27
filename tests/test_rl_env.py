from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from core.rl.config import RLConfig
from core.rl.env import BenchmarkEnv, EnvState, _env_step_jnp, _sample_training_state


class _FixedNoise:
    def __init__(self, sample):
        self.sample = jnp.asarray(sample, dtype=jnp.float32)

    def sample_jax(self, rng, shape=()):
        del rng
        return jnp.broadcast_to(self.sample, tuple(shape) + self.sample.shape)


class _MazeModel:
    n = 4
    state_variables = ["x_pos", "x_vel", "y_pos", "y_vel"]
    partition = {
        "boundary": np.array([[0.0, -1.0, 0.0, -1.0], [10.0, 1.0, 10.0, 1.0]]),
        "number_per_dim": np.array([10, 2, 10, 2]),
    }
    goal = np.array([[[0.0, -1.0, 8.0, -1.0], [2.0, 1.0, 10.0, 1.0]]])
    critical = np.array([[[0.0, -1.0, 4.0, -1.0], [8.0, 1.0, 6.0, 1.0]]])
    x0 = np.array([1.0, 0.0, 1.0, 0.0])
    uMin = [-1.0, -1.0]
    uMax = [1.0, 1.0]
    num_actions = [5, 5]
    noise = _FixedNoise([0.0, 0.0, 0.0, 0.0])

    @staticmethod
    def step(state, action, noise):
        del action
        return state + noise


def _config(**overrides):
    values = dict(
        max_steps=20,
        goal_reward=10.0,
        unsafe_penalty=-10.0,
        out_of_bounds_penalty=-10.0,
        distance_reward=1.0,
        per_step_reward=0.0,
    )
    values.update(overrides)
    return RLConfig(**values)


class TestRLEnvironment(unittest.TestCase):
    def test_training_actions_use_downstream_control_grid(self):
        env = BenchmarkEnv(_MazeModel(), _config())
        actions = jnp.array([[0.74, -0.26], [-4.0, 3.0]])

        snapped = env.discretize_action(actions)

        np.testing.assert_allclose(
            np.asarray(snapped), np.array([[0.5, -0.5], [-1.0, 1.0]])
        )

    def test_curriculum_expands_from_near_goal_to_full_safe_domain(self):
        env = BenchmarkEnv(
            _MazeModel(),
            _config(use_curriculum=True, curriculum_initial_fraction=0.05),
        )
        keys = jax.random.split(jax.random.PRNGKey(3), 256)
        near = jax.vmap(lambda key: _sample_training_state(key, env, 0.0))(keys)
        full = jax.vmap(lambda key: _sample_training_state(key, env, 1.0))(keys)
        goal_pos = env.goal_center_jnp[env.position_dims_jnp]
        near_distance = jnp.linalg.norm(near[:, env.position_dims_jnp] - goal_pos, axis=1)
        full_distance = jnp.linalg.norm(full[:, env.position_dims_jnp] - goal_pos, axis=1)

        self.assertLess(float(jnp.mean(near_distance)), float(jnp.mean(full_distance)))

    def test_curriculum_does_not_change_nonterminal_reward(self):
        env = BenchmarkEnv(_MazeModel(), _config(use_curriculum=True, distance_reward=0.0))
        state = jnp.array([3.0, 1.0, 2.0, -1.0])
        env_state = EnvState(state=state, steps=jnp.array(0), prev_dist=env.distance_to_goal(state))

        _, _, reward, done, _ = _env_step_jnp(
            jax.random.PRNGKey(0), env_state, jnp.zeros(2), env
        )

        self.assertFalse(bool(done))
        self.assertAlmostEqual(float(reward), 0.0)

    def test_training_noise_is_not_scaled(self):
        model = _MazeModel()
        model.noise = _FixedNoise([0.25, 0.0, 0.25, 0.0])
        env = BenchmarkEnv(model, _config(distance_reward=0.0))
        state = jnp.array([3.0, 0.0, 2.0, 0.0])
        env_state = EnvState(state=state, steps=jnp.array(0), prev_dist=env.distance_to_goal(state))

        next_obs, _, _, done, _ = _env_step_jnp(
            jax.random.PRNGKey(0), env_state, jnp.zeros(2), env
        )

        self.assertFalse(bool(done))
        np.testing.assert_allclose(np.asarray(next_obs), np.array([3.25, 0.0, 2.25, 0.0]))


if __name__ == "__main__":
    unittest.main()
