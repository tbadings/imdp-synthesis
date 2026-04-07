import argparse
import os
import unittest

import jax
import numpy as np

import benchmarks
from core.abstraction.probability_intervals import compute_probability_intervals
from core.abstraction.forward_reachability import RectangularForward
from core.abstraction.imdp import IMDP, RVI_JAX
from core.abstraction.model import parse_nonlinear_model
from core.abstraction.partition import RectangularPartition


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


def _compute_initial_value(model_name: str):
    args = _make_hardcoded_args(model_name=model_name)
    base_model = getattr(benchmarks, model_name)(args)
    model = parse_nonlinear_model(base_model)

    partition = RectangularPartition(model=model)
    actions = RectangularForward(args=args, partition=partition, model=model)

    P_full, S_id, A_id, P_absorbing = compute_probability_intervals(
        args=args,
        model=model,
        partition=partition,
        actions=actions,
        vectorized=True,
    )

    imdp = IMDP(
        partition=partition,
        states=np.array(partition.regions["idxs"]),
        actions_inputs=actions.id_to_input,
        x0=model.x0,
        goal_regions=np.array(partition.goal["bools"]),
        critical_regions=np.array(partition.critical["bools"]),
        P_full=P_full,
        S_id=S_id,
        A_id=A_id,
        P_absorbing=P_absorbing,
    )

    s0 = partition.x2state(model.x0)[0]
    V, _, _ = RVI_JAX(
        args=args,
        imdp=imdp,
        s0=s0,
        max_iterations=10000,
        epsilon=1e-6,
        RND_SWEEPS=True,
        BATCH_SIZE=1000,
        policy_iteration=args.policy_iteration,
    )
    return int(s0), V


class TestBenchmarkInitialValues(unittest.TestCase):
    def test_initial_state_optimal_values(self) -> None:
        expected = {
            "Dubins3D": (467, 0.867012),
            "Test1D": (2, 0.999601),
            "DoubleIntegrator": (212, 0.998698),
            "MountainCar": (4235, 0.991068),
            "Pendulum": (4975, 0.992158),
        }

        for model_name, (expected_s0, expected_value) in expected.items():
            with self.subTest(model=model_name):
                s0, V = _compute_initial_value(model_name)
                self.assertEqual(s0, expected_s0)
                self.assertTrue(
                    np.isclose(float(V[s0]), expected_value, atol=1e-3),
                    msg=f"Expected V[{expected_s0}] ~= {expected_value:.6f}, got {float(V[s0]):.6f}",
                )


if __name__ == "__main__":
    unittest.main()