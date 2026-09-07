import hashlib
import math
from decimal import localcontext
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from core.options import parse_arguments
from core.rl.config import RLConfig, resolve_rl_config
from core.rl.initializers import _fixed_qr, deterministic_orthogonal
from core.rl.policy import ActorCritic


class TestDeterministicOrthogonal(unittest.TestCase):
    def test_rectangular_and_square_orthogonality_with_original_gains(self):
        for shape in ((4, 32), (32, 4), (16, 16), (1, 7), (7, 1), (1, 1)):
            for gain in (math.sqrt(2), 0.01, 1.0):
                with self.subTest(shape=shape, gain=gain):
                    weights = np.asarray(deterministic_orthogonal(gain)(
                        jax.random.PRNGKey(7), shape
                    ))
                    self.assertEqual(weights.shape, shape)
                    self.assertEqual(weights.dtype, np.dtype("float32"))
                    weights = weights.astype(np.float64) / gain
                    basis = weights if shape[0] >= shape[1] else weights.T
                    gram = np.einsum("ki,kj->ij", basis, basis)
                    np.testing.assert_allclose(gram, np.eye(min(shape)), atol=1e-7, rtol=1e-7)

    def test_fixed_qr_matches_positive_diagonal_reference(self):
        # Independent LAPACK reference checks the reflector order and signs,
        # beyond orthogonality (which alone would also accept a constant basis).
        matrix = np.array([[2., 4., -1.], [-3., 2., 7.], [1., 3., 9.], [5., -1., 2.]])
        q, r = np.linalg.qr(matrix)
        np.testing.assert_allclose(_fixed_qr(matrix), q * np.sign(np.diag(r)), atol=2e-15)

    def test_key_reproducibility_and_decimal_context_independence(self):
        init = deterministic_orthogonal()
        expected = np.asarray(init(jax.random.PRNGKey(7), (8, 5)))
        with localcontext() as context:
            context.prec = 6
            actual = np.asarray(init(jax.random.key(7), (8, 5)))
        self.assertEqual(expected.tobytes(), actual.tobytes())
        self.assertNotEqual(expected.tobytes(), np.asarray(init(jax.random.key(8), (8, 5))).tobytes())

    def test_versioned_float32_fingerprint(self):
        # This fingerprint is a portability contract for the versioned sampler
        # and QR. Run this same test on both machines when investigating drift.
        weights = np.asarray(deterministic_orthogonal(math.sqrt(2))(
            jax.random.PRNGKey(7), (4, 7)
        ), dtype="<f4")
        self.assertEqual(hashlib.sha256(weights.tobytes()).hexdigest(),
                         "eac5e1a2919795b2aca4856a4caf9a8f463c0f1a27766dee92f6e6e9b75811d3")

        # Exercise the 256x256 shape that showed native-QR drift on M3/M5.
        weights = np.asarray(deterministic_orthogonal(math.sqrt(2))(
            jax.random.PRNGKey(7), (256, 256)
        ), dtype="<f4")
        self.assertEqual(hashlib.sha256(weights.tobytes()).hexdigest(),
                         "18560d4870a160838126793ddde67dfc7fe5fd6c89c931f0f9ac11de1ff18ac9")
        basis = weights.astype(np.float64) / math.sqrt(2)
        np.testing.assert_allclose(np.einsum("ki,kj->ij", basis, basis), np.eye(256),
                                   atol=1e-7, rtol=1e-7)

    def test_jit_and_vmap_match_eager_initialization(self):
        init = deterministic_orthogonal(0.01)
        keys = jax.random.split(jax.random.key(42), 3)
        initialize = lambda key: init(key, (9, 4))
        eager = np.stack([np.asarray(initialize(key)) for key in keys])
        compiled = np.stack([np.asarray(jax.jit(initialize)(key)) for key in keys])
        batched = np.asarray(jax.jit(jax.vmap(initialize))(keys))
        self.assertEqual(eager.tobytes(), compiled.tobytes())
        self.assertEqual(eager.tobytes(), batched.tobytes())

    def test_float64_output(self):
        with jax.enable_x64():
            weights = np.asarray(deterministic_orthogonal()(
                jax.random.key(0), (12, 5), jnp.float64
            ))
            self.assertEqual(weights.dtype, np.dtype("float64"))
            np.testing.assert_allclose(np.einsum("ki,kj->ij", weights, weights), np.eye(5), atol=2e-15)

    def test_invalid_shape_dtype_and_scale(self):
        init = deterministic_orthogonal()
        key = jax.random.key(0)
        for shape in ((3,), (2, 3, 4), (0, 3)):
            with self.assertRaisesRegex(ValueError, "nonempty 2-D"):
                init(key, shape)
        with self.assertRaisesRegex(ValueError, "float32 and float64"):
            init(key, (3, 4), jnp.int32)
        with self.assertRaisesRegex(ValueError, "finite scale"):
            deterministic_orthogonal(float("nan"))

    def test_policy_gains_jitted_init_and_training_gradients(self):
        policy = ActorCritic(action_dim=2, pi_arch=(16, 16), vf_arch=(16, 16),
                             init_method="deterministic_orthogonal")
        observations = jnp.ones((3, 4))
        key = jax.random.key(42)
        params = policy.init(key, observations)
        compiled_params = jax.jit(policy.init)(key, observations)
        for eager, compiled in zip(jax.tree.leaves(params), jax.tree.leaves(compiled_params)):
            self.assertEqual(np.asarray(eager).tobytes(), np.asarray(compiled).tobytes())
        gains = (math.sqrt(2), math.sqrt(2), 0.01, math.sqrt(2), math.sqrt(2), 1.0)
        for index, gain in enumerate(gains):
            layer = params["params"][f"Dense_{index}"]
            weights = np.asarray(layer["kernel"], dtype=np.float64) / gain
            rows, columns = weights.shape
            basis = weights if rows >= columns else weights.T
            gram = np.einsum("ki,kj->ij", basis, basis)
            np.testing.assert_allclose(gram, np.eye(min(rows, columns)), atol=1e-7, rtol=1e-7)
            np.testing.assert_array_equal(layer["bias"], 0)
        np.testing.assert_array_equal(params["params"]["log_std"], 0)

        def loss(variables):
            mean, log_std, value = policy.apply(variables, observations)
            return jnp.mean((mean - 1)**2) + jnp.mean((value - 1)**2) + jnp.sum(log_std)

        value, gradients = jax.jit(jax.value_and_grad(loss))(params)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(gradients)))
        self.assertGreater(float(jnp.linalg.norm(gradients["params"]["Dense_0"]["kernel"])), 0)

    def test_config_default_and_cli_override(self):
        self.assertEqual(RLConfig().init_method, "orthogonal")
        self.assertEqual(ActorCritic(action_dim=2).init_method, "orthogonal")
        model = SimpleNamespace(rl_config=RLConfig(init_method="uniform"))
        args = parse_arguments(["--init_method", "deterministic_orthogonal"])
        self.assertEqual(resolve_rl_config(model, args).init_method, "deterministic_orthogonal")
        self.assertEqual(resolve_rl_config(model, parse_arguments([])).init_method, "uniform")


if __name__ == "__main__":
    unittest.main()
