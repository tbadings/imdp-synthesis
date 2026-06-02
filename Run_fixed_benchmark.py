"""
Fixed-arguments launcher for experimentation.
"""

import subprocess
import sys
from pathlib import Path

def main() -> None:
    root = Path(__file__).resolve().parent
    runfile = root / "Main_IMDP.py"

    fixed_args = [
        "--model",
        "Drone6D_small",
        "--batch_size",
        "1000",
        "--noise_distr",
        "normal",
        "--solver",
        "jax",
    ]

    cmd = [sys.executable, str(runfile), *fixed_args]
    subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
