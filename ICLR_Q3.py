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

def config_CollisionAvoidance() -> list[str]:
    return [
        "--model",
        "CollisionAvoidance",
        "--solver",
        "jax",
        "--satprob",
        "0.99",
        "--seed",
        "0",
    ]

def config_TripleIntegrator() -> list[str]:
    return [
        "--model",
        "TripleIntegrator",
        "--solver",
        "jax",
        "--satprob",
        "0.99",
        "--seed",
        "0",
    ]

def config_PlanarRobot() -> list[str]:
    return [
        "--model",
        "PlanarRobot",
        "--solver",
        "jax",
        "--satprob",
        "0.99",
        "--seed",
        "0",
    ]

def config_Drone4D_logRASM() -> list[str]:
    return [
        "--model",
        "Drone4D_logRASM",
        "--solver",
        "jax",
        "--satprob",
        "0.99",
        "--seed",
        "0",
    ]

if __name__ == "__main__":
    # =========================================================================
    # logRASM Benchmarks
    # =========================================================================
    # CollisionAvoidance
    run_fixed_SVMDP(args = ["--model","CollisionAvoidance", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","CollisionAvoidance", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","CollisionAvoidance", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","CollisionAvoidance", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","CollisionAvoidance", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo"])
    # TripleIntegrator
    run_fixed_SVMDP(args = ["--model","TripleIntegrator", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","TripleIntegrator", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","TripleIntegrator", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","TripleIntegrator", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","TripleIntegrator", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo"])
    # PlanarRobot
    run_fixed_SVMDP(args = ["--model","PlanarRobot", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","PlanarRobot", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","PlanarRobot", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","PlanarRobot", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","PlanarRobot", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo"])
    # Drone4D (logRASM)
    run_fixed_SVMDP(args = ["--model","Drone4D_logRASM", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","Drone4D_logRASM", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","Drone4D_logRASM", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","Drone4D_logRASM", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo"])
    run_fixed_SVMDP(args = ["--model","Drone4D_logRASM", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo"])
