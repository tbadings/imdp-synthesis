"""
Fixed-arguments launcher for experimentation.

This script invokes RunFile.py with a hardcoded argument list.
"""

import subprocess
import sys
from pathlib import Path

def main() -> None:
    root = Path(__file__).resolve().parent
    runfile = root / "Main_IMDP.py"

    fixed_args = [
        "--model",
        "Dubins3D",
        "--batch_size",
        "1000",
        "--noise_distr",
        "normal",
        "--solver",
        "storm",
    ]

    cmd = [sys.executable, str(runfile), *fixed_args]
    subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
