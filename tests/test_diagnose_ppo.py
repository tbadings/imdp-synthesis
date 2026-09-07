"""Comparison must detect small discrepancies without hiding invalid results."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from diagnose_ppo import SCHEMA, compare_array, compare_runs


class TestDiagnosticComparison(unittest.TestCase):
    def test_small_rollout_difference_is_located_even_with_loose_tolerance(self):
        a = np.ones((5, 2, 3), dtype=np.float32)
        b = a.copy()
        b[3, 1, 2] += 1e-5
        result = compare_array(a, b, atol=1e-3, rtol=0)
        self.assertFalse(result["exact"])
        self.assertEqual(result["different_elements"], 1)
        self.assertEqual(result["outside_tolerance"], 0)
        self.assertEqual(result["first_different_index"], [3, 1, 2])
        self.assertAlmostEqual(result["max_abs"], 1e-5, delta=2e-8)

    def test_signed_zero_and_nonfinite_values(self):
        result = compare_array(np.array(0.0), np.array(-0.0), 0, 0)
        self.assertFalse(result["exact"])
        self.assertEqual(result["max_abs"], 0)
        self.assertEqual(result["first_different_index"], [])
        invalid = np.array([np.nan, np.inf, -np.inf])
        result = compare_array(invalid, invalid.copy(), 0, 0)
        self.assertTrue(result["exact"])
        self.assertEqual(result["nonfinite_elements"], 3)
        self.assertEqual(result["outside_tolerance"], 3)

    def test_large_integer_difference_and_incompatible_arrays(self):
        result = compare_array(np.array([2**63], dtype=np.uint64),
                               np.array([2**63 + 1], dtype=np.uint64), 1e6, 1)
        self.assertEqual(result["max_abs"], 1)
        self.assertEqual(result["outside_tolerance"], 1)
        self.assertFalse(compare_array(np.zeros(2), np.zeros(3), 0, 0)["compatible"])
        self.assertFalse(compare_array(np.zeros(2, dtype=np.float32), np.zeros(2), 0, 0)["compatible"])

    def test_archives_report_missing_arrays_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                out = root / name
                out.mkdir()
                (out / "metadata.json").write_text(json.dumps({"schema": SCHEMA, "mode": "run"}))
            np.savez(root / "a/arrays.npz", common=np.array([np.inf]), missing=np.ones(1))
            np.savez(root / "b/arrays.npz", common=np.array([np.inf]))
            options = argparse.Namespace(a=root / "a", b=root / "b", atol=0, rtol=0,
                                         top=5, report=root / "comparison.json")
            with redirect_stdout(io.StringIO()):
                status = compare_runs(options)
            self.assertEqual(status, 1)
            report = json.loads(options.report.read_text())
            self.assertEqual(report["changed_arrays"], 2)
            self.assertEqual(report["arrays"]["missing"]["missing_from"], "b")
            self.assertEqual(report["arrays"]["common"]["nonfinite_elements"], 1)


if __name__ == "__main__":
    unittest.main()
