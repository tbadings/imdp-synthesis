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
    #     "--total_timesteps",
    #     "200000",
    #     "--noise_distr",
    #     "gaussian",
    #     # "--no-policy_iteration",
    #     # "--shrink_frs",
    #     # "0",
    #     "--RL_actions_per_state",
    #     "100",
    #     "--max_steps",
    #     "200",
    #     "--eval_episodes",
    #     "100",
    #     "--goal_reward",
    #     "5",
    #     "--unsafe_penalty",
    #     "-5",
    #     "--out_of_bounds_penalty",
    #     "-5",
    #     "--distance_reward",
    #     "1.0",
    #     "--per_step_reward",
    #     "-0.1",
    # ]

    # fixed_args = [
    #     "--model",
    #     "CartPole",
    #     "--satprob",
    #     "0.99",
    #     # "--batch_size",
    #     # "1000",
    #     # "--noise_distr",
    #     # "normal",
    #     "--solver",
    #     "jax",
    #     "--total_timesteps",
    #     "500000",
    #     "--noise_distr",
    #     "gaussian",
    #     # "--no-policy_iteration",
    #     # "--shrink_frs",
    #     # "0",
    #     "--RL_actions_per_state",
    #     "5",
    #     "--max_steps",
    #     "200",
    #     "--eval_episodes",
    #     "10000",
    #     "--goal_reward",
    #     "5",
    #     "--unsafe_penalty",
    #     "-5",
    #     "--out_of_bounds_penalty",
    #     "-5",
    #     "--distance_reward",
    #     "1.0",
    #     "--save_checkpoint",
    #     # "--ent_coef",
    #     # "0.005",
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
    #     "100000",
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
    #     "Dubins4D",
    #     # "--batch_size",
    #     # "1000",
    #     # "--noise_distr",
    #     # "normal",
    #     "--solver",
    #     "jax",
    #     # "--eval_episodes",
    #     # "5000",
    #     "--total_timesteps",
    #     "500000",
    #     "--noise_distr",
    #     "gaussian",
    #     # "--no-policy_iteration",
    #     # "--shrink_frs",
    #     # "0",
    #     "--RL_actions_per_state",
    #     "25",
    # ]

    fixed_args = [
        "--model",
        "Drone4D",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
        # "--eval_episodes",
        # "5000",
        "--total_timesteps",
        "100000",
        "--noise_distr",
        "gaussian",
        # "--no-policy_iteration",
        "--RL_actions_per_state",
        "9"
    ]

    # fixed_args = [
    #     "--model",
    #     "Drone4D_battery",
    #     "--satprob",
    #     "0.95",
    #     # "--batch_size",
    #     # "1000",
    #     "--solver",
    #     "jax",
    #     "--eval_episodes",
    #     "1000",
    #     "--total_timesteps",
    #     "200000",
    #     "--noise_distr",
    #     "gaussian",
    #     # "--no-policy_iteration",
    #     # "--shrink_frs",
    #     # "0",
    #     "--RL_actions_per_state",
    #     "27",
    #     # "--save_checkpoint",
    #     # "--load_checkpoint",
    #     # "output/2026-06-30_22-27-54_Drone6D/checkpoint.pkl",
    # ]

    # fixed_args = [
    #     "--model",
    #     "Drone6D_battery",
    #     "--satprob",
    #     "0.95",
    #     # "--batch_size",
    #     # "1000",
    #     "--solver",
    #     "jax",
    #     "--eval_episodes",
    #     "1000",
    #     "--total_timesteps",
    #     "400000",
    #     "--noise_distr",
    #     "gaussian",
    #     # "--no-policy_iteration",
    #     # "--shrink_frs",
    #     # "0",
    #     "--RL_actions_per_state",
    #     "27",
    #     # "--save_checkpoint",
    #     # "--load_checkpoint",
    #     # "output/2026-06-30_22-27-54_Drone6D/checkpoint.pkl",
    # ]


    cmd = [sys.executable, str(runfile), *fixed_args]
    subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
