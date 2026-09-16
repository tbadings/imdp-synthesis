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
        # "--dense",
        "--satprob",
        "0.99",
        "--seed",
        "0",
    ]

def config_CartPole() -> list[str]:
    return [
        "--model",
        "CartPole",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
        # "--dense",
        "--satprob",
        "0.99",
        "--seed",
        "0",
    ]

def config_Pendulum() -> list[str]:
    return [
        "--model",
        "Pendulum",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
        # "--dense",
        "--satprob",
        "0.99",
        "--seed",
        "0",
    ]

def config_Dubins3D() -> list[str]:
    return [
        "--model",
        "Dubins3D",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
        # "--dense",
        "--satprob",
        "0.99",
        "--seed",
        "0",
    ]

def config_Dubins4D() -> list[str]:
    return [
        "--model",
        "Dubins4D",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
        # "--dense",
        "--satprob",
        "0.99",
        "--seed",
        "0",
    ]

def config_Drone4D() -> list[str]:
    return [
        "--model",
        "Drone4D",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
        # "--dense",
        "--satprob",
        "0.99",
        "--seed",
        "0",
    ]

def config_Drone6D_small() -> list[str]:
    return [
        "--model",
        "Drone6D_small",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
        # "--dense",
        "--satprob",
        "0.99",
        "--seed",
        "0",
    ]

def config_Drone6D() -> list[str]:
    return [
        "--model",
        "Drone6D",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
        # "--dense",
        "--satprob",
        "0.99",
        "--seed",
        "0",
    ]

def config_Drone4D_battery() -> list[str]:
    return [
        "--model",
        "Drone4D_battery",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
        # "--dense",
        "--satprob",
        "0.99",
        "--seed",
        "0",
    ]

# To run a particular benchmark, simply change the argument in the function call below
if __name__ == "__main__":
    # run_fixed_SVMDP(args = config_MountainCar())
    # run_fixed_SVMDP(args = config_Pendulum())
    # run_fixed_SVMDP(args = config_CartPole())
    # run_fixed_SVMDP(args = config_Dubins3D())
    # run_fixed_SVMDP(args = config_Dubins4D())
    # run_fixed_SVMDP(args = config_Drone4D())
    # run_fixed_SVMDP(args = config_Drone6D())
    # run_fixed_SVMDP(args = config_Drone4D_battery())


    run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--dense"])
    run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","sac"])
    run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac"])
    run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","sac"])
