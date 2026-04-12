"""
Fixed-arguments launcher for experimentation.

This script invokes RunFile.py with a hardcoded argument list.
"""

import subprocess
import sys
from pathlib import Path

# sys.argv = ['RunFile.py', '--model', 'Dubins3D', '--batch_size', '1000', '--noise_distr', 'normal']
# sys.argv = ['RunFile.py', '--model', 'Dubins4D', '--batch_size', '1000']
# sys.argv = ['RunFile.py', '--model', 'Pendulum', '--batch_size', '1000']
# sys.argv = ['RunFile.py', '--model', 'MountainCar', '--batch_size', '1000', '--plot_title']
# sys.argv = ['RunFile.py', '--model', 'DoubleIntegrator', '--batch_size', '30000', '--plot_title']
# sys.argv = ['RunFile.py', '--model', 'Drone3D_small', '--batch_size', '100', '--plot_title']
# sys.argv = ['RunFile.py', '--model', 'Drone3D', '--batch_size', '10000', '--plot_title']
# sys.argv = ['RunFile.py', '--model', 'Drone2D', '--batch_size', '1000', '--plot_title']

def main() -> None:
    root = Path(__file__).resolve().parent
    runfile = root / "RunFile.py"

    fixed_args = [
        "--model",
        "Dubins4D",
        "--batch_size",
        "1000",
        "--noise_distr",
        "normal",
    ]

    cmd = [sys.executable, str(runfile), *fixed_args]
    subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
