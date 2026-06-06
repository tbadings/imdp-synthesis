import argparse
import itertools
import os
import unittest

import jax
import numpy as np

import benchmarks
from benchmarks.dynamics import setmath
from core.abstraction.model import parse_nonlinear_model


os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax.config.update("jax_default_matmul_precision", "high")
jax.config.update("jax_platform_name", "cpu")


def _make_args() -> argparse.Namespace:
    return argparse.Namespace(
        model="Cartpole",
        model_version=0,
        noise_distr="gaussian",
    )


class TestCartpoleBenchmark(unittest.TestCase):
    def test_cartpole_is_registered(self) -> None:
        model = benchmarks.create_model(_make_args())

        self.assertEqual(model.n, 4)
        self.assertEqual(model.p, 1)
        self.assertEqual(model.num_actions, [7])

    def test_step_matches_gymnasium_cartpole_dynamics(self) -> None:
        model = benchmarks.Cartpole(_make_args())
        state = np.array([0.0, 0.0, 0.05, 0.0])
        action = np.array([10.0])
        noise = np.zeros(4)

        successor = model.step(state, action, noise)

        np.testing.assert_allclose(
            successor,
            np.array([0.0, 0.19437055, 0.05, -0.27649758]),
            atol=1e-8,
        )

    def test_step_set_contains_vertex_successors(self) -> None:
        model = parse_nonlinear_model(benchmarks.Cartpole(_make_args()))
        state_min = np.array([-0.1, -0.1, 0.04, -0.1])
        state_max = np.array([0.1, 0.1, 0.06, 0.1])
        action_min = np.array([-10.0])
        action_max = np.array([10.0])

        actual_min, actual_max = model.step_set(state_min, state_max, action_min, action_max)
        actual_min = np.asarray(actual_min, dtype=float)
        actual_max = np.asarray(actual_max, dtype=float)

        state_vertices = np.asarray(setmath.box2vertices(state_min, state_max), dtype=float)
        action_vertices = np.asarray(setmath.box2vertices(action_min, action_max), dtype=float)
        for state, action in itertools.product(state_vertices, action_vertices):
            successor = model.step(state, action, np.zeros(4))
            self.assertTrue(np.all(successor >= actual_min - 1e-7))
            self.assertTrue(np.all(successor <= actual_max + 1e-7))


if __name__ == "__main__":
    unittest.main()
