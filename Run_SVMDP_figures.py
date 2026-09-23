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

    # Figure for in appendix
    run_fixed_SVMDP(args = ["--model","Drone4D", "--solver","jax", "--satprob","0.99", "--damping","0.1", "--seed","0", "--algo","ppo", "--eval_episodes", "100", "--plot_SA_tube", "--save_checkpoint"])