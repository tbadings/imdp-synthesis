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
    result = subprocess.run(cmd, cwd=root)
    if result.returncode != 0:
        # Negative code = killed by a signal (-9 is typical for an OS out-of-memory kill)
        print(f"!! Run failed (exit code {result.returncode}): {' '.join(args)}", file=sys.stderr)

# To run a particular benchmark, simply change the argument in the function call below
if __name__ == "__main__":

    # Drone4D_x0set
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        v = str(v)
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--solver","jax", "--satprob","0.99", "--dense", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo", "--model_version",v])
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        v = str(v)
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","sac", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","sac", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","sac", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Drone4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","sac", "--model_version",v])

    # Dubins4D_x0set
    for v in [1, 2, 3, 4]:
        v = str(v)
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--solver","jax", "--satprob","0.99", "--dense", "--model_version", v])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo", "--model_version", v])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo", "--model_version",v])
    for v in [1, 2, 3, 4]:
        v = str(v)
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","sac", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","sac", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","sac", "--model_version",v])
        run_fixed_SVMDP(args = ["--model","Dubins4D_x0set", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","sac", "--model_version",v])

    # CartPole_x0set
    # for v in [2, 5, 8]:
    #         v = str(v)
    #         run_fixed_SVMDP(args = ["--model","Cartpole_x0set", "--solver","jax", "--satprob","0.99", "--dense", "--model_version", v])
    #         run_fixed_SVMDP(args = ["--model","Cartpole_x0set", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","ppo", "--model_version", v])
    #         run_fixed_SVMDP(args = ["--model","Cartpole_x0set", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","ppo", "--model_version",v])
    #         run_fixed_SVMDP(args = ["--model","Cartpole_x0set", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","ppo", "--model_version",v])
    #         run_fixed_SVMDP(args = ["--model","Cartpole_x0set", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","ppo", "--model_version",v])
    #         run_fixed_SVMDP(args = ["--model","Cartpole_x0set", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","ppo", "--model_version",v])
    #         run_fixed_SVMDP(args = ["--model","Cartpole_x0set", "--solver","jax", "--satprob","0.99", "--seed","0", "--algo","sac", "--model_version",v])
    #         run_fixed_SVMDP(args = ["--model","Cartpole_x0set", "--solver","jax", "--satprob","0.99", "--seed","1", "--algo","sac", "--model_version",v])
    #         run_fixed_SVMDP(args = ["--model","Cartpole_x0set", "--solver","jax", "--satprob","0.99", "--seed","2", "--algo","sac", "--model_version",v])
    #         run_fixed_SVMDP(args = ["--model","Cartpole_x0set", "--solver","jax", "--satprob","0.99", "--seed","3", "--algo","sac", "--model_version",v])
    #         run_fixed_SVMDP(args = ["--model","Cartpole_x0set", "--solver","jax", "--satprob","0.99", "--seed","4", "--algo","sac", "--model_version",v])