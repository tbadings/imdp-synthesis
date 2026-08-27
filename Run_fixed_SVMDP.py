"""
Fixed-arguments launcher for experimentation.

Reward terms are not set here: each benchmark declares its own in `self.rl_reward`
(see benchmarks/*.py). Passing --goal_reward / --unsafe_penalty /
--out_of_bounds_penalty / --per_step_reward / --distance_reward below would
override the benchmark's value.
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
        # "--noise_distr",
        # "normal",
        "--solver",
        "jax",
        "--total_timesteps",
        "200000",
        "--noise_distr",
        "gaussian",
        # "--no-policy_iteration",
        # "--shrink_frs",
        # "0",
        "--RL_actions_per_state",
        "100",
        "--max_steps",
        "200",
        "--eval_episodes",
        "100",
        "--tube_method",
        "smart"
    ]

def config_CartPole() -> list[str]:
    return [
        "--model",
        "CartPole",
        "--satprob",
        "0.99",
        # "--batch_size",
        # "1000",
        # "--noise_distr",
        # "normal",
        "--solver",
        "jax",
        "--total_timesteps",
        "500000",
        "--noise_distr",
        "gaussian",
        # "--no-policy_iteration",
        # "--shrink_frs",
        # "0",
        "--RL_actions_per_state",
        "5",
        "--max_steps",
        "200",
        "--eval_episodes",
        "10000",
        "--save_checkpoint",
        # "--ent_coef",
        # "0.005",
    ]

def config_Dubins3D() -> list[str]:
    return [
        "--model",
        "Dubins3D",
        # "--batch_size",
        # "1000",
        # "--noise_distr",
        # "normal",
        "--solver",
        "jax",
        # "--eval_episodes",
        # "5000",
        "--total_timesteps",
        "100000",
        "--noise_distr",
        "gaussian",
        # "--no-policy_iteration",
        # "--shrink_frs",
        # "0",
        "--RL_actions_per_state",
        "100"
    ]

def config_Dubins4D() -> list[str]:
    return [
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
        "500000",
        "--noise_distr",
        "gaussian",
        # "--no-policy_iteration",
        # "--shrink_frs",
        # "0",
        "--RL_actions_per_state",
        "25",
    ]

def config_Drone4D() -> list[str]:
    # The reward lives on the benchmark itself now (Drone4D.rl_reward); passing any of
    # --goal_reward / --unsafe_penalty / --out_of_bounds_penalty / --per_step_reward /
    # --distance_reward here would override it.
    #
    # Drone4D is an S-shaped maze: from x0 the drone flies right through the gap in the lower
    # wall, left through the gap in the upper wall, and only then right to the goal, dodging a
    # lane obstacle in each of the outer bands. The reward balance is load-bearing: the terminal
    # penalties must exceed the cost of surviving without reaching the goal,
    # (|per_step_reward| + norm(distance_reward)) / (1 - gamma). Below that, flying into a wall
    # or out of the domain is the cheapest way out of the bottom band, and PPO learns to do it.
    return [
        "--model",
        "Drone4D",
        "--solver",
        "jax",
        "--eval_episodes",
        "1000",
        # The two lane obstacles make this map hard enough that the budget matters: at 8M the
        # policy settles for hovering on roughly one seed in three, at 4M on every seed. None
        # of the reward knobs fixed that -- only more training did.
        "--total_timesteps",
        "16000000",
        "--n_envs",
        "256",
        # Short training episodes are load-bearing here: at --max_steps 64 or 128 the policy
        # settles for hovering and never reaches the goal. --eval_steps keeps the rollouts long
        # enough to actually get there (~60 steps from x0).
        "--max_steps",
        "32",
        "--eval_steps",
        "200",
        "--noise_distr",
        "gaussian",
        "--RL_actions_per_state",
        "9",
    ]

    # fixed_args = [
    #     "--model",
    #     "Drone4D",
    #     "--satprob",
    #     "0.95",
    #     # "--batch_size",
    #     # "1000",
    #     "--solver",
    #     "jax",
    #     "--eval_episodes",
    #     "100",
    #     "--total_timesteps",
    #     "20000",
    #     "--noise_distr",
    #     "gaussian",
    #     # "--no-policy_iteration",
    #     # "--shrink_frs",
    #     # "0",
    #     "--RL_actions_per_state",
    #     "9",
    #     # "--save_checkpoint",
    #     # "--load_checkpoint",
    #     # "output/2026-06-30_22-27-54_Drone6D/checkpoint.pkl",
    #     "--tube_method",
    #     "smart"
    # ]

def config_Drone6D_small() -> list[str]:
    return [
        "--model",
        "Drone6D_small",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
        "--eval_episodes",
        "1000",
        "--total_timesteps",
        "100000",
        "--noise_distr",
        "gaussian",
        # "--no-policy_iteration",
        "--RL_actions_per_state",
        "27",
        "--satprob",
        "0.95",
        # "--tube_method",
        # "smart",
    ]

def config_Drone6D() -> list[str]:
    return [
        "--model",
        "Drone6D",
        # "--batch_size",
        # "1000",
        "--solver",
        "jax",
        "--eval_episodes",
        "1000",
        "--total_timesteps",
        "1000000",
        "--noise_distr",
        "gaussian",
        # "--no-policy_iteration",
        "--RL_actions_per_state",
        "27",
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
        "--eval_episodes",
        "1000",
        "--total_timesteps",
        "400000",
        "--noise_distr",
        "gaussian",
        # "--no-policy_iteration",
        # "--shrink_frs",
        # "0",
        "--RL_actions_per_state",
        "27",
        # "--save_checkpoint",
        # "--load_checkpoint",
        # "output/2026-06-30_22-27-54_Drone6D/checkpoint.pkl",
    ]

# To run a particular benchmark, simply change the argument in the function call below
if __name__ == "__main__":
    # run_fixed_SVMDP(args = config_MountainCar())
    # run_fixed_SVMDP(args = config_CartPole())
    # run_fixed_SVMDP(args = config_Dubins3D())
    # run_fixed_SVMDP(args = config_Dubins4D())
    run_fixed_SVMDP(args = config_Drone4D())
    # run_fixed_SVMDP(args = config_Drone6D_small())
    # run_fixed_SVMDP(args = config_Drone6D())
    # run_fixed_SVMDP(args = config_Drone6D_battery())