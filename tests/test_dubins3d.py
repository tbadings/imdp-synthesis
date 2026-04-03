import argparse
import os
import unittest

import jax
import numpy as np

import benchmarks
from core.Gaussian_probabilities import compute_probability_intervals
from core.forward_reachability import RectangularForward
from core.imdp import IMDP, RVI_JAX
from core.model import parse_nonlinear_model
from core.partition import RectangularPartition


os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax.config.update("jax_default_matmul_precision", "high")
jax.config.update("jax_platform_name", "cpu")


def _make_hardcoded_dubins3d_args() -> argparse.Namespace:
    args = argparse.Namespace(
        debug=False,
        seed=0,
        decimals=4,
        pAbs_min=0.0001,
        model="Dubins3D",
        model_version=0,
        gpu=False,
        gpu_rvi=False,
        policy_iteration=True,
        mode="fori_loop",
        batch_size=100,
        plot_grid=False,
        plot_title=False,
        plot_ticks=False,
    )
    args.floatprecision = np.float32
    args.rvi_device = jax.devices("cpu")[0]
    np.random.seed(args.seed)
    args.jax_key = jax.random.PRNGKey(args.seed)
    return args


class TestDubins3DInitialValue(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        args = _make_hardcoded_dubins3d_args()
        base_model = benchmarks.Dubins3D(args)
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

        cls.s0 = s0
        cls.V = V

    def test_initial_state_optimal_value(self) -> None:
        self.assertEqual(self.s0, 467)
        self.assertTrue(
            np.isclose(float(self.V[self.s0]), 0.867012, atol=1e-3),
            msg=f"Expected V[467] ~= 0.867012, got {float(self.V[self.s0]):.6f}",
        )


if __name__ == "__main__":
    unittest.main()