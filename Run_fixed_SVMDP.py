"""
Fixed-arguments launcher for experimentation.
"""

import subprocess
import sys
from pathlib import Path

def main() -> None:
    root = Path(__file__).resolve().parent
    runfile = root / "Main_SVMDP.py"

    # fixed_args = [
    #     "--model",
    #     "MountainCar",
    #     # "--batch_size",
    #     # "1000",
    #     # "--noise_distr",
    #     # "normal",
    #     "--solver",
    #     "jax",
    #     # "--eval_episodes",
    #     # "5000",
    #     "--total_timesteps",
    #     "200000",
    #     "--noise_distr",
    #     "gaussian",
    #     # "--no-policy_iteration",
    #     # "--shrink_frs",
    #     # "0",
    #     "--RL_actions_per_state",
    #     "100"
    # ]

    # fixed_args = [
    #     "--model",
    #     "Dubins3D",
    #     # "--batch_size",
    #     # "1000",
    #     # "--noise_distr",
    #     # "normal",
    #     "--solver",
    #     "jax",
    #     # "--eval_episodes",
    #     # "5000",
    #     "--total_timesteps",
    #     "200000",
    #     "--noise_distr",
    #     "gaussian",
    #     # "--no-policy_iteration",
    #     # "--shrink_frs",
    #     # "0",
    #     "--RL_actions_per_state",
    #     "100"
    # ]

    fixed_args = [
        "--model",
        "Dubins4D",
        # "--batch_size",
        # "1000",
        # "--noise_distr",
        # "normal",
        "--solver",
        "jax",
        # "--eval_episodes",
        # "5000",
        "--total_timesteps",
        "200000",
        "--noise_distr",
        "gaussian",
        # "--no-policy_iteration",
        # "--shrink_frs",
        # "0",
        "--RL_actions_per_state",
        "25",
    ]

    # fixed_args = [
    #     "--model",
    #     "Drone4D",
    #     # "--batch_size",
    #     # "1000",
    #     "--solver",
    #     "jax",
    #     # "--eval_episodes",
    #     # "5000",
    #     "--total_timesteps",
    #     "100000",
    #     "--noise_distr",
    #     "gaussian",
    #     # "--no-policy_iteration",
    #     # "--shrink_frs",
    #     # "0",
    #     "--RL_actions_per_state",
    #     "27"
    # ]

    # fixed_args = [
    #     "--model",
    #     "Drone6D",
    #     # "--batch_size",
    #     # "1000",
    #     "--solver",
    #     "jax",
    #     "--eval_episodes",
    #     "1000",
    #     "--total_timesteps",
    #     "500000",
    #     "--noise_distr",
    #     "gaussian",
    #     # "--no-policy_iteration",
    #     # "--shrink_frs",
    #     # "0",
    #     "--RL_actions_per_state",
    #     "27",
    #     "--save_checkpoint"
    # ]

    # fixed_args = [
    #     "--model",
    #     "Drone6D_small",
    #     # "--batch_size",
    #     # "1000",
    #     "--solver",
    #     "jax",
    #     "--eval_episodes",
    #     "1000",
    #     "--total_timesteps",
    #     "500000",
    #     "--noise_distr",
    #     "gaussian",
    #     # "--no-policy_iteration",
    #     # "--shrink_frs",
    #     # "0",
    #     "--RL_actions_per_state",
    #     "27",
    #     "--load_checkpoint",
    #     "output/2026-06-30_10-21-14_Drone6D_small/checkpoint.pkl"
    # ]

    cmd = [sys.executable, str(runfile), *fixed_args]
    subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
