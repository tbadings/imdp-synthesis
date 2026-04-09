"""Run one named test module or the full tests suite from the tests directory."""

import argparse
import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# Ensure project-level imports (e.g., `import benchmarks`) work regardless of where
# this runner is invoked from.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _available_test_files() -> dict[str, Path]:
    return {path.name: path for path in sorted(TESTS_DIR.glob("test_*.py"))}


def _resolve_test_pattern(target: str, available: dict[str, Path]) -> str:
    normalized = Path(target).name

    candidates = [normalized]
    if not normalized.endswith(".py"):
        candidates.append(f"{normalized}.py")
    if not normalized.startswith("test_"):
        candidates.append(f"test_{normalized}")
        if not normalized.endswith(".py"):
            candidates.append(f"test_{normalized}.py")

    for candidate in candidates:
        if candidate in available:
            return candidate

    available_names = ", ".join(available)
    raise ValueError(f"Unknown test target '{target}'. Available tests: {available_names}")


def _build_suite(targets: list[str]) -> unittest.TestSuite:
    available = _available_test_files()
    loader = unittest.defaultTestLoader

    if not targets or targets == ["all"]:
        return loader.discover(str(TESTS_DIR), pattern="test_*.py")

    suite = unittest.TestSuite()
    for target in targets:
        pattern = _resolve_test_pattern(target, available)
        suite.addTests(loader.discover(str(TESTS_DIR), pattern=pattern))
    return suite


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one or all tests from the tests directory.",
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Test target(s) to run. Use 'all' or omit to run every test.",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available test modules and exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Run tests with higher verbosity.",
    )
    args = parser.parse_args()

    available = _available_test_files()
    if args.list:
        for name in available:
            print(name)
        return 0

    try:
        suite = _build_suite(args.tests)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())