"""
Fixed-arguments launcher for experimentation.
"""

import subprocess
import sys
from pathlib import Path

def main() -> None:
    root = Path(__file__).resolve().parent
    runfile = root / "Main_SVMDP.py"

    fixed_args = [
        "--model",
        "MountainCar",
        # "--batch_size",
        # "1000",
        # "--noise_distr",
        # "normal",
        "--solver",
        "storm",
        # "--eval_episodes",
        # "5000",
        # "--total_timesteps",
        # "100000",
        "--noise_distr",
        "triangular",
    ]

    cmd = [sys.executable, str(runfile), *fixed_args]
    subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
