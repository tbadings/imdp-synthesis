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

# Table 1 budgets: the larger PPO/SAC average for states and for choices.
if __name__ == "__main__":
    # MountainCar
    # run_fixed_SVMDP(args = ["--model", "MountainCar", "--solver", "jax", "--satprob", "0.99", "--dense", "--partition_states", 36_052])
    # run_fixed_SVMDP(args = ["--model", "MountainCar", "--solver", "jax", "--satprob", "0.99", "--dense", "--partition_choices", 108_157])

    # # Pendulum
    # run_fixed_SVMDP(args = ["--model", "Pendulum", "--solver", "jax", "--satprob", "0.99", "--dense", "--partition_states", 63_384])
    # run_fixed_SVMDP(args = ["--model", "Pendulum", "--solver", "jax", "--satprob", "0.99", "--dense", "--partition_choices", 316_919])

    # # Dubins3D
    # run_fixed_SVMDP(args = ["--model", "Dubins3D", "--solver", "jax", "--satprob", "0.99", "--dense", "--partition_states", 279_393])
    # run_fixed_SVMDP(args = ["--model", "Dubins3D", "--solver", "jax", "--satprob", "0.99", "--dense", "--partition_choices", 2_514_539])

    # CartPole
    # run_fixed_SVMDP(args = ["--model", "CartPole", "--solver", "jax", "--satprob", "0.99", "--dense", "--partition_states", 3_381_422])
    # run_fixed_SVMDP(args = ["--model", "CartPole", "--solver", "jax", "--satprob", "0.99", "--dense", "--partition_choices", 16_907_112])

    # Dubins4D
    # run_fixed_SVMDP(args = ["--model", "Dubins4D", "--solver", "jax", "--satprob", "0.99", "--dense", "--partition_states", 2_129_803])
    # run_fixed_SVMDP(args = ["--model", "Dubins4D", "--solver", "jax", "--satprob", "0.99", "--dense", "--partition_choices", 19_168_229])

    # Drone4D
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--dense", "--partition_states", 93_797])
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--dense", "--partition_choices", 844_173])

    # Drone4D (+damping)
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--dense", "--partition_states", 94_092])
    run_fixed_SVMDP(args = ["--model", "Drone4D", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--dense", "--partition_choices", 846_826])

    # Drone4D (+battery and damping). DroneDynamics_battery currently ignores --damping.
    run_fixed_SVMDP(args = ["--model", "Drone4D_battery", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--dense", "--partition_states", 617_218])
    run_fixed_SVMDP(args = ["--model", "Drone4D_battery", "--solver", "jax", "--satprob", "0.99", "--damping", "0.1", "--dense", "--partition_choices", 15_430_440])

    # Drone6D has no PPO/SAC state or choice averages in Table 1.
