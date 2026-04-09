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


def _make_hardcoded_args(model_name: str) -> argparse.Namespace:
    args = argparse.Namespace(
        debug=False,
        seed=0,
        decimals=4,
        pAbs_min=0.0001,
        model=model_name,
        model_version=0,
        noise_distr="gaussian",
        gpu=False,
        gpu_rvi=False,
        policy_iteration=True,
        mode="fori_loop",
        batch_size=100,
        plot_grid=False,
        plot_title=False,
        plot_ticks=False,
        frs_batch_size=1000,
    )
    args.floatprecision = np.float32
    args.rvi_device = jax.devices("cpu")[0]
    np.random.seed(args.seed)
    args.jax_key = jax.random.PRNGKey(args.seed)
    return args


def _exact_reachable_vertices(model, state_min, state_max, action_min, action_max):
    state_vertices = np.asarray(setmath.box2vertices(state_min, state_max), dtype=float)

    action_min = np.maximum(np.asarray(action_min, dtype=float), np.asarray(model.uMin, dtype=float))
    action_max = np.minimum(np.asarray(action_max, dtype=float), np.asarray(model.uMax, dtype=float))
    action_vertices = np.asarray(setmath.box2vertices(action_min, action_max), dtype=float)

    successors = []
    for state_vertex, action_vertex in itertools.product(state_vertices, action_vertices):
        successors.append(np.asarray(model.A @ state_vertex + model.B @ action_vertex, dtype=float))

    return np.vstack(successors)


class TestDroneStepSet(unittest.TestCase):
    def test_step_set_returns_exact_interval_hull_for_drone_models(self) -> None:
        cases = {
            "Drone2D": {
                "state_min": np.array([1.2, 0.8, -0.5, 1.5]),
                "state_max": np.array([-0.4, -1.2, 2.0, -0.1]),
                "action_min": np.array([4.0, -1.5]),
                "action_max": np.array([-2.0, 5.0]),
            },
            "Drone3D": {
                "state_min": np.array([2.0, -0.2, 1.5, 0.6, -1.0, 0.4]),
                "state_max": np.array([0.5, -1.0, -0.5, -0.3, 1.2, -0.8]),
                "action_min": np.array([5.0, -1.5, 1.0]),
                "action_max": np.array([-2.0, 6.0, -4.0]),
            },
        }

        for model_name, case in cases.items():
            with self.subTest(model=model_name):
                model = parse_nonlinear_model(
                    getattr(benchmarks, model_name)(_make_hardcoded_args(model_name))
                )

                actual_min, actual_max = model.step_set(
                    case["state_min"],
                    case["state_max"],
                    case["action_min"],
                    case["action_max"],
                )
                actual_min = np.asarray(actual_min, dtype=float)
                actual_max = np.asarray(actual_max, dtype=float)

                exact_successors = _exact_reachable_vertices(
                    model,
                    case["state_min"],
                    case["state_max"],
                    case["action_min"],
                    case["action_max"],
                )
                expected_min = np.min(exact_successors, axis=0)
                expected_max = np.max(exact_successors, axis=0)

                print(f"\n[{model_name}] exact_successors=\n{exact_successors}")
                print(f"[{model_name}] expected_min={expected_min}")
                print(f"[{model_name}] expected_max={expected_max}")
                print(f"[{model_name}] actual_min={actual_min}")
                print(f"[{model_name}] actual_max={actual_max}")

                np.testing.assert_allclose(actual_min, expected_min, atol=1e-6)
                np.testing.assert_allclose(actual_max, expected_max, atol=1e-6)

                for dim in range(model.n):
                    self.assertTrue(np.any(np.isclose(exact_successors[:, dim], actual_min[dim], atol=1e-6)))
                    self.assertTrue(np.any(np.isclose(exact_successors[:, dim], actual_max[dim], atol=1e-6)))


if __name__ == "__main__":
    unittest.main()