import unittest

from core.options import parse_arguments


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


if __name__ == "__main__":
    unittest.main()
