from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from core.rl.config import RLConfig
from core.rl.env import _in_boxes_jnp
from core.rl.evaluation import _sample_initial_states, evaluate_policy


class TestRLEvaluationInitialStates(unittest.TestCase):
    def test_random_initial_states_span_domain_and_are_safe(self) -> None:
        env = SimpleNamespace(
            obs_dim=2,
            obs_low_jnp=jnp.array([-10.0, -10.0]),
            obs_high_jnp=jnp.array([10.0, 10.0]),
            reset_low_jnp=jnp.array([-1.0, -1.0]),
            reset_high_jnp=jnp.array([1.0, 1.0]),
            critical_jnp=jnp.array([[[-2.0, -2.0], [2.0, 2.0]]]),
            goal_jnp=jnp.array([[[8.0, 8.0], [10.0, 10.0]]]),
        )
        keys = jax.random.split(jax.random.PRNGKey(7), 64)

        local_states = np.asarray(_sample_initial_states(env, keys, random_across_domain=False))
        random_states = np.asarray(_sample_initial_states(env, keys, random_across_domain=True))

        self.assertTrue(np.all(local_states >= -1.0))
        self.assertTrue(np.all(local_states <= 1.0))
        self.assertTrue(np.any((random_states < -1.0) | (random_states > 1.0)))
        self.assertFalse(np.any(np.asarray(_in_boxes_jnp(jnp.asarray(random_states), env.critical_jnp))))
        self.assertFalse(np.any(np.asarray(_in_boxes_jnp(jnp.asarray(random_states), env.goal_jnp))))

    @patch("core.rl.evaluation.plot_rl_trajectories")
    @patch("core.rl.evaluation._build_batch_evaluator")
    @patch("core.rl.evaluation._sample_initial_states")
    def test_plot_uses_random_starts_without_changing_evaluation_tube(
        self,
        sample_initial_states,
        build_batch_evaluator,
        plot_rl_trajectories,
    ) -> None:
        local_initial_states = jnp.array([[1.1], [2.1]], dtype=jnp.float32)
        random_initial_states = jnp.array([[7.1], [8.1]], dtype=jnp.float32)
        sample_initial_states.side_effect = [local_initial_states, random_initial_states]

        def fake_evaluator(params, discrete_actions, initial_states, step_keys):
            del params, discrete_actions, step_keys
            initial_states = np.asarray(initial_states)
            next_states = np.repeat(initial_states[:, None, :], 2, axis=1)
            was_done = np.ones((len(initial_states), 2), dtype=bool)
            reached_goal = np.zeros(len(initial_states), dtype=bool)
            return initial_states, next_states, was_done, reached_goal

        build_batch_evaluator.return_value = fake_evaluator
        env = SimpleNamespace(
            obs_low=np.array([0.0]),
            bin_widths=np.array([1.0]),
            number_per_dim=np.array([10]),
        )
        model = SimpleNamespace(partition={"number_per_dim": np.array([10])})
        cfg = RLConfig(
            max_steps=2,
            goal_reward=1.0,
            unsafe_penalty=-1.0,
            out_of_bounds_penalty=-1.0,
            distance_reward=0.0,
            per_step_reward=0.0,
        )

        _, visited_cells, _ = evaluate_policy(
            actor_critic=object(),
            params=None,
            base_model=model,
            env=env,
            cfg=cfg,
            episodes=2,
            dims=[0],
            args=SimpleNamespace(output_dir="unused"),
            seed=3,
        )

        self.assertEqual(visited_cells, {(1,), (2,)})
        plotted_trajectories = plot_rl_trajectories.call_args.args[2]
        np.testing.assert_allclose(
            np.array([trace[0, 0] for trace in plotted_trajectories]),
            np.array([7.1, 8.1]),
        )
        self.assertFalse(sample_initial_states.call_args_list[0].kwargs["random_across_domain"])
        self.assertTrue(sample_initial_states.call_args_list[1].kwargs["random_across_domain"])


if __name__ == "__main__":
    unittest.main()
