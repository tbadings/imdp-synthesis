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

# Table 1 state budgets: the larger PPO/SAC average for each benchmark.
# The second run also fixes the number of enabled RL actions per state.
if __name__ == "__main__":
    # MountainCar
    run_fixed_SVMDP(args = ["--model", "MountainCar", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_states", 36_052])
    run_fixed_SVMDP(args = ["--model", "MountainCar", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_states", 36_052, "--fix_num_actions", 3])

    # Pendulum
    run_fixed_SVMDP(args = ["--model", "Pendulum", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_states", 63_384])
    run_fixed_SVMDP(args = ["--model", "Pendulum", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_states", 63_384, "--fix_num_actions", 5])

    # # Dubins3D
    run_fixed_SVMDP(args = ["--model", "Dubins3D", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_states", 279_393])
    run_fixed_SVMDP(args = ["--model", "Dubins3D", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_states", 279_393, "--fix_num_actions", 9])

    # CartPole
    run_fixed_SVMDP(args = ["--model", "CartPole", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_states", 3_381_422])
    run_fixed_SVMDP(args = ["--model", "CartPole", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_states", 3_381_422, "--fix_num_actions", 5])

    # Dubins4D
    run_fixed_SVMDP(args = ["--model", "Dubins4D", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_states", 2_129_803])
    run_fixed_SVMDP(args = ["--model", "Dubins4D", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_states", 2_129_803, "--fix_num_actions", 9])

    # Drone4D
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_states", 93_797])
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--dense", "--fix_num_states", 93_797, "--fix_num_actions", 9])

    # Drone4D (+damping)
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--dense", "--fix_num_states", 94_092])
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--dense", "--fix_num_states", 94_092, "--fix_num_actions", 9])

    # Drone4D (+battery and damping)
    run_fixed_SVMDP(args = ["--model", "Drone4D_battery", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--dense", "--fix_num_states", 617_218])
    run_fixed_SVMDP(args = ["--model", "Drone4D_battery", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--dense", "--fix_num_states", 617_218, "--fix_num_actions", 25])

    # Drone 6D
    run_fixed_SVMDP(args = ["--model", "Drone6D", "--solver", "jax", "--satprob","0.99", "--dense", "--fix_num_states", 1_000_000])
    run_fixed_SVMDP(args = ["--model", "Drone6D", "--solver", "jax", "--satprob","0.99", "--dense", "--fix_num_states", 1_000_000, "--fix_num_actions", 27])