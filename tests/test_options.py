import unittest

from core.options import parse_arguments
from core.utils import DPStagnationCheck


class TestOptions(unittest.TestCase):
    def test_noise_normal_alias(self) -> None:
        args = parse_arguments(["--noise_distr", "normal"])
        self.assertEqual(args.noise_distr, "gaussian")

    def test_log_level_default(self) -> None:
        args = parse_arguments([])
        self.assertEqual(args.log_level, "INFO")

    def test_batch_size_must_be_positive(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--batch_size", "0"])

    def test_decimals_must_be_nonnegative(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--decimals", "-1"])

    def test_pabs_min_must_be_probability(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--pAbs_min", "1.5"])

    def test_dp_terminate_defaults_disabled(self) -> None:
        args = parse_arguments([])
        self.assertEqual(args.dp_terminate_window, 0)
        self.assertEqual(args.dp_terminate_improvement, 0.0)
        self.assertEqual(args.dp_terminate_min_iterations, 0)

    def test_dp_terminate_must_be_nonnegative(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--dp_terminate_window", "-1"])
        with self.assertRaises(SystemExit):
            parse_arguments(["--dp_terminate_improvement", "-0.1"])
        with self.assertRaises(SystemExit):
            parse_arguments(["--dp_terminate_min_iterations", "-1"])


class TestDPStagnationCheck(unittest.TestCase):
    def test_disabled_if_either_is_zero(self) -> None:
        for iterations, improvement in [(0, 0.1), (5, 0.0), (0, 0.0)]:
            check = DPStagnationCheck(iterations, improvement)
            self.assertFalse(check.enabled)
            self.assertFalse(any(check.update(0.5) for _ in range(100)))

    def test_terminates_after_window_without_improvement(self) -> None:
        check = DPStagnationCheck(3, 0.01)
        results = [check.update(v) for v in [0.0, 0.1, 0.2, 0.3, 0.305, 0.308, 0.309]]
        self.assertEqual(results, [False] * 6 + [True])

    def test_min_iterations_delays_termination(self) -> None:
        # Value stuck at zero: without a minimum, terminates right after the first full window (5 iterations)
        check = DPStagnationCheck(5, 0.001)
        results = [check.update(0.0) for _ in range(1 + 10)]  # initial value + 10 iterations
        self.assertEqual(results.index(True), 5)

        # With a minimum of 10 iterations, it cannot terminate before the 10th full iteration
        check = DPStagnationCheck(5, 0.001, min_iterations=10)
        results = [check.update(0.0) for _ in range(1 + 12)]
        self.assertEqual(results.index(True), 10)

    def test_min_iterations_uses_sliding_window(self) -> None:
        # Improvement during the first iterations does not prevent stagnation being detected at the minimum
        check = DPStagnationCheck(3, 0.01, min_iterations=6)
        values = [0.0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3]
        results = [check.update(v) for v in values]
        self.assertEqual(results, [False] * 6 + [True, True])


if __name__ == "__main__":
    unittest.main()
