"""
Fixed-arguments launcher for experimentation.

The RL settings (PPO training, reward function, and the tube around the rollouts) live in
each benchmark's `rl_config` in `benchmarks/`; only what is not benchmark-specific is passed
here. Any RL option can still be overridden per run by adding e.g. "--total_timesteps",
"1000" below, which takes precedence over the benchmark's `rl_config`.
"""

import subprocess
import sys
from pathlib import Path

def run_fixed_SVMDP(args: list[str]) -> None:
    root = Path(__file__).resolve().parent
    runfile = root / "Main_SVMDP.py"

    cmd = [sys.executable, str(runfile), *args]
    subprocess.run(cmd, check=True, cwd=root)

def config_MountainCar() -> list[str]:
    return [
        "--model",
        "MountainCar",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
    ]

def config_CartPole() -> list[str]:
    return [
        "--model",
        "CartPole",
        "--satprob",
        "0.99",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
    ]

def config_Dubins3D() -> list[str]:
    return [
        "--model",
        "Dubins3D",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
    ]

def config_Dubins4D() -> list[str]:
    return [
        "--model",
        "Dubins4D",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
    ]

def config_Drone4D() -> list[str]:
    return [
        "--model",
        "Drone4D",
        "--solver",
        "jax",
    ]

def config_Drone6D_small() -> list[str]:
    return [
        "--model",
        "Drone6D_small",
        "--satprob",
        "0.95",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
    ]

def config_Drone6D() -> list[str]:
    return [
        "--model",
        "Drone6D",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
    ]

def config_Drone6D_battery() -> list[str]:
    return [
        "--model",
        "Drone6D_battery",
        "--satprob",
        "0.95",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
    ]

# To run a particular benchmark, simply change the argument in the function call below
if __name__ == "__main__":
    # run_fixed_SVMDP(args = config_MountainCar())
    run_fixed_SVMDP(args = config_CartPole())
    # run_fixed_SVMDP(args = config_Dubins3D())
    # run_fixed_SVMDP(args = config_Dubins4D())
    # run_fixed_SVMDP(args = config_Drone4D())
    # run_fixed_SVMDP(args = config_Drone6D_small())
    # run_fixed_SVMDP(args = config_Drone6D())
    # run_fixed_SVMDP(args = config_Drone6D_battery())
