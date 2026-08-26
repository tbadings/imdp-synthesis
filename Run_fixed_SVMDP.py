"""
Fixed-arguments launcher for experimentation.
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
        "--goal_reward",
        "5",
        "--unsafe_penalty",
        "-20",
        "--out_of_bounds_penalty",
        "-20",
        # distance_reward is now a cost on the distance itself, not on the change in distance,
        # so it needs a much smaller gain: at 1.0 the cost of surviving (1.1 / 0.01 = 110)
        # dwarfs any penalty and the car drives itself out of the domain instead.
        "--distance_reward",
        "0.05",
        "--per_step_reward",
        "-0.1",
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
        "2000000",
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
        "--goal_reward",
        "5",
        "--unsafe_penalty",
        "-20",
        "--out_of_bounds_penalty",
        "-20",
        # See the MountainCar note: distance_reward is a cost on the distance now, not on its change.
        "--distance_reward",
        "0.05",
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
    # Drone4D is an S-shaped maze: from x0 the drone has to fly right through the gap in
    # the lower wall, all the way left through the gap in the upper wall, and only then
    # right to the goal. Two parts of this config are load-bearing:
    #
    #  * The terminal penalties (-35) must exceed the cost of surviving without reaching
    #    the goal, (|per_step_reward| + |distance_reward|) / (1 - gamma) = 0.3/0.01 = 30.
    #    Below that threshold the best available behaviour in the bottom band is to end
    #    the episode on purpose, and PPO reliably learns to fly straight out of the domain.
    #  * The distance cost gives the far bottom-left corner a gradient to follow. With a
    #    flat per-step cost that corner is a plateau, and whether it gets solved is
    #    seed-dependent.
    #
    # --max_steps is the *training* truncation (short episodes reset often, which spreads
    # the data over the whole state space); --eval_steps is the rollout horizon, which has
    # to stay long enough to reach the goal (~60 steps from x0).
    return [
        "--model",
        "Drone4D",
        "--solver",
        "jax",
        "--eval_episodes",
        "1000",
        "--total_timesteps",
        "4000000",
        "--n_envs",
        "256",
        "--max_steps",
        "32",
        "--eval_steps",
        "200",
        "--goal_reward",
        "5",
        "--unsafe_penalty",
        "-35",
        "--out_of_bounds_penalty",
        "-35",
        "--per_step_reward",
        "-0.1",
        "--distance_reward",
        "0.2",
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