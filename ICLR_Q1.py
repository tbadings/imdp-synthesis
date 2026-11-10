"""
Fixed-arguments launcher for experimentation.

The RL settings (PPO training, reward function, and the tube around the rollouts) live in
each benchmark's `rl_config` in `benchmarks/`; only what is not benchmark-specific is passed
here. Any RL option can still be overridden per run by adding e.g. "--total_timesteps",
"1000" below, which takes precedence over the benchmark's `rl_config`.
RRT settings are supplied directly to a single run_fixed_SVMDP call via its rrt argument.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def run_fixed_SVMDP(args: list[str], *, rrt=None) -> None:
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env['IMDP_RRT_SETTINGS'] = json.dumps(rrt or {})
    subprocess.run([sys.executable, str(root / 'Main_SVMDP.py'), *args],
                   check=True, cwd=root, env=env)


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
    for tube_radius in ((4, 2, 4, 2), (5, 3, 5, 3), (6, 4, 6, 4), (7, 5, 7, 5), (8, 6, 8, 6)):
        for seed in range(5):
            run_fixed_SVMDP(
                args=["--model", "Drone4D", "--rrt", "--solver", "jax", "--satprob", "0.99", "--seed", str(seed)],
                rrt=dict(
                    iterations=15000,
                    step_horizon=1,
                    control_samples=16,
                    parent_candidates=16,
                    goal_bias=0.1,
                    time_weight=0.01,
                    tube_radius=tube_radius,
                    clearance=(2,),
                ),
            )
    # MountainCar
    # run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--dense"])
    # run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","MountainCar", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","sac"])

    # Pendulum
    # run_fixed_SVMDP(args = ["--model","Pendulum", "--solver","jax", "--satprob","0.99", "--dense"])
    # run_fixed_SVMDP(args = ["--model","Pendulum", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Pendulum", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Pendulum", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Pendulum", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Pendulum", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Pendulum", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Pendulum", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Pendulum", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Pendulum", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Pendulum", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","sac"])

    # CartPole
    # run_fixed_SVMDP(args = ["--model","CartPole", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","CartPole", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","CartPole", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","CartPole", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","CartPole", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","CartPole", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","CartPole", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","CartPole", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","CartPole", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","CartPole", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","sac"])

    # Dubins3D
    # run_fixed_SVMDP(args = ["--model","Dubins3D", "--solver","jax", "--satprob","0.99", "--dense"])
    # run_fixed_SVMDP(args = ["--model","Dubins3D", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Dubins3D", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Dubins3D", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Dubins3D", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Dubins3D", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Dubins3D", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Dubins3D", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Dubins3D", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Dubins3D", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Dubins3D", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","sac"])

    # Dubins4D
    # run_fixed_SVMDP(args = ["--model","Dubins4D", "--solver","jax", "--satprob","0.99", "--dense"])
    # run_fixed_SVMDP(args = ["--model","Dubins4D", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Dubins4D", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Dubins4D", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Dubins4D", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Dubins4D", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Dubins4D", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Dubins4D", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Dubins4D", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Dubins4D", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Dubins4D", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","sac"])

    # Drone4D
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--dense"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","sac"])

    # Drone4D
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--dense"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","0", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","1", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","2", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","3", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","4", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","0", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","1", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","2", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","3", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","4", "--algo","sac"])

    # Drone6D
    # run_fixed_SVMDP(args = ["--model","Drone6D", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone6D", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone6D", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone6D", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone6D", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone6D", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone6D", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone6D", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone6D", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone6D", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","sac"])

    # Drone4D_battery
    # run_fixed_SVMDP(args = ["--model","Drone4D_battery", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--dense"])
    # run_fixed_SVMDP(args = ["--model","Drone4D_battery", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","0", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D_battery", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","1", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D_battery", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","2", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D_battery", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","3", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D_battery", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","4", "--algo","ppo"])
    # run_fixed_SVMDP(args = ["--model","Drone4D_battery", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","0", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone4D_battery", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","1", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone4D_battery", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","2", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone4D_battery", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","3", "--algo","sac"])
    # run_fixed_SVMDP(args = ["--model","Drone4D_battery", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","4", "--algo","sac"])
