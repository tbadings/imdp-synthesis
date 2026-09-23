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
    run_fixed_SVMDP(args = ["--model", "MountainCar", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_choices", 108_157])
    run_fixed_SVMDP(args = ["--model", "MountainCar", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_choices", 108_157, "--fix_num_actions", 3])

    # Pendulum
    run_fixed_SVMDP(args = ["--model", "Pendulum", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_choices", 316_919])
    run_fixed_SVMDP(args = ["--model", "Pendulum", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_choices", 316_919, "--fix_num_actions", 5])

    # Dubins3D
    run_fixed_SVMDP(args = ["--model", "Dubins3D", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_choices", 2_514_539])
    run_fixed_SVMDP(args = ["--model", "Dubins3D", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_choices", 2_514_539, "--fix_num_actions", 9])

    # CartPole
    run_fixed_SVMDP(args = ["--model", "CartPole", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_choices", 16_907_112])
    run_fixed_SVMDP(args = ["--model", "CartPole", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_choices", 16_907_112, "--fix_num_actions", 5])

    # Dubins4D
    run_fixed_SVMDP(args = ["--model", "Dubins4D", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_choices", 19_168_229])
    run_fixed_SVMDP(args = ["--model", "Dubins4D", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_choices", 19_168_229, "--fix_num_actions", 9])

    # Drone4D
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_choices", 844_173])
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_choices", 844_173, "--fix_num_actions", 9])

    # Drone4D (+damping)
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--dense", "--fix_num_choices", 846_826])
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--dense", "--fix_num_choices", 846_826, "--fix_num_actions", 9])

    # Drone4D (+battery and damping)
    run_fixed_SVMDP(args = ["--model", "Drone4D_battery", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--dense", "--fix_num_choices", 15_430_440])
    run_fixed_SVMDP(args = ["--model", "Drone4D_battery", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--dense", "--fix_num_choices", 15_430_440, "--fix_num_actions", 25])

    # Drone6D has no PPO/SAC choice averages in Table 1.

    pass
