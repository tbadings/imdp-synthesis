import argparse
import os
import unittest

import jax
import numpy as np

import benchmarks
from core.abstraction.forward_reachability import RectangularForward
from core.abstraction.model import parse_nonlinear_model
from core.abstraction.partition import RectangularPartition
from core.abstraction.probability_intervals import compute_probability_intervals


os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax.config.update("jax_default_matmul_precision", "high")
jax.config.update("jax_platform_name", "cpu")


def _make_hardcoded_test1d_args(noise_distr: str) -> argparse.Namespace:
    args = argparse.Namespace(
        debug=False,
        seed=0,
        decimals=4,
        pAbs_min=0.0001,
        model="Test1D",
        model_version=0,
        noise_distr=noise_distr,
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


class TestTest1DProbabilityIntervals(unittest.TestCase):
    def _compute_intervals(self, noise_distr: str):
        args = _make_hardcoded_test1d_args(noise_distr=noise_distr)
        base_model = benchmarks.Test1D(args)
        model = parse_nonlinear_model(base_model)

        partition = RectangularPartition(model=model)
        actions = RectangularForward(args=args, partition=partition, model=model)

        print(actions.frs_lb[0,1])
        print(actions.frs_ub[0,1])
        print(actions.frs_idx_lb[0,1])
        print(actions.max_slice)

        P_full, S_id, A_id, P_absorbing = compute_probability_intervals(
            args=args,
            model=model,
            partition=partition,
            actions=actions,
            vectorized=True,
        )
        return P_full, S_id, A_id, P_absorbing

    def test_state_action_probability_intervals_for_both_noise_models(self) -> None:
        expected_by_noise = {
            "triangular": {
                "successors": np.array([0, 1, 9], dtype=int),
                "intervals": np.array(
                    [
                        [0.5, 0.75],
                        [0.0001, 0.5],
                        [0.0001, 0.5],
                    ],
                    dtype=np.float32,
                ),
                "absorbing": np.array([0.0001, 0.0001], dtype=np.float32),
            },
            "gaussian": {
                "successors": np.array([0, 1, 9], dtype=int),
                "intervals": np.array(
                    [
                        [0.4998, 0.9876],
                        [0.0001, 0.5002],
                        [0.0001, 0.4998],
                    ],
                    dtype=np.float32,
                ),
                "absorbing": np.array([0.0001, 0.0001], dtype=np.float32),
            },
        }

        state = 0
        action_label = 1

        for noise_distr, expected in expected_by_noise.items():
            with self.subTest(noise_distr=noise_distr):
                P_full, S_id, A_id, P_absorbing = self._compute_intervals(noise_distr)

                action_idx_candidates = np.where(A_id[state] == action_label)[0]
                self.assertGreater(
                    len(action_idx_candidates),
                    0,
                    msg=f"Action label {action_label} not enabled in state {state} for {noise_distr}",
                )
                action_idx = int(action_idx_candidates[0])

                successors = np.array(S_id[state][action_idx])
                intervals = np.array(P_full[state][action_idx])
                absorbing = np.array(P_absorbing[state][action_idx])

                print(
                    f"\n[Test1D/{noise_distr}] state={state}, action_label={action_label}, "
                    f"successors={successors.tolist()}, intervals={intervals.tolist()}, "
                    f"absorbing={absorbing.tolist()}"
                )

                np.testing.assert_array_equal(
                    successors,
                    expected["successors"],
                )
                np.testing.assert_allclose(
                    intervals,
                    expected["intervals"],
                    atol=1e-4,
                )
                np.testing.assert_allclose(
                    absorbing,
                    expected["absorbing"],
                    atol=1e-4,
                )

                total_lb = float(np.sum(intervals[:, 0]) + absorbing[0])
                total_ub = float(np.sum(intervals[:, 1]) + absorbing[1])
                self.assertLessEqual(total_lb, 1.0 + 1e-4)
                self.assertGreaterEqual(total_ub, 1.0 - 1e-4)


if __name__ == "__main__":
    unittest.main()
