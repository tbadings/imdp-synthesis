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

def run_fixed_SVMDP(args: list[str | int]) -> None:
    root = Path(__file__).resolve().parent
    runfile = root / "Main_SVMDP.py"

    cmd = [sys.executable, str(runfile), *(str(arg) for arg in args)]
    subprocess.run(cmd, check=True, cwd=root)

# Table 1 choice budgets: the larger PPO/SAC average for each benchmark.
# The second run also fixes the number of enabled RL actions per state.
if __name__ == "__main__":

    # MountainCar
    run_fixed_SVMDP(args = ["--model", "MountainCar", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 110_000])
    run_fixed_SVMDP(args = ["--model", "MountainCar", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 110_000, "--fix_num_actions", 3])

    # Pendulum
    run_fixed_SVMDP(args = ["--model", "Pendulum", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 320_000])
    run_fixed_SVMDP(args = ["--model", "Pendulum", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 320_000, "--fix_num_actions", 5])

    # Dubins3D
    run_fixed_SVMDP(args = ["--model", "Dubins3D", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 2_500_000])
    run_fixed_SVMDP(args = ["--model", "Dubins3D", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 2_500_000, "--fix_num_actions", 9])

    # CartPole
    run_fixed_SVMDP(args = ["--model", "CartPole", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 17_000_000])
    run_fixed_SVMDP(args = ["--model", "CartPole", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 17_000_000, "--fix_num_actions", 5])

    # Dubins4D
    run_fixed_SVMDP(args = ["--model", "Dubins4D", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 19_000_000])
    run_fixed_SVMDP(args = ["--model", "Dubins4D", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 19_000_000, "--fix_num_actions", 9])

    # Drone4D
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 790_000])
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 790_000, "--fix_num_actions", 9])

    # Drone4D (+damping)
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--seed", "0", "--dense", "--fix_num_choices", 850_000])
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--seed", "0", "--dense", "--fix_num_choices", 850_000, "--fix_num_actions", 9])

    # Drone4D (+battery and damping)
    run_fixed_SVMDP(args = ["--model", "Drone4D_battery", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--seed", "0", "--dense", "--fix_num_choices", 16_000_000])
    run_fixed_SVMDP(args = ["--model", "Drone4D_battery", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--seed", "0", "--dense", "--fix_num_choices", 16_000_000, "--fix_num_actions", 25])

    # Drone6D
    run_fixed_SVMDP(args = ["--model", "Drone6D", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 89_000_000])
    run_fixed_SVMDP(args = ["--model", "Drone6D", "--solver", "jax", "--satprob", "0.99", "--seed", "0", "--dense", "--fix_num_choices", 89_000_000, "--fix_num_actions", 27])
