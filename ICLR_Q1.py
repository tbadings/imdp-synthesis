"""
Fixed-arguments launcher for experimentation.

The RL settings (PPO training, reward function, and the tube around the rollouts) live in
each benchmark's `rl_config` in `benchmarks/`; only what is not benchmark-specific is passed
here. Any RL option can still be overridden for all runs by adding e.g. "--total_timesteps",
"1000" to EXTRA_ARGS below, which takes precedence over the benchmark's `rl_config`.
"""

import subprocess
import sys
from pathlib import Path

def run_fixed_SVMDP(args: list[str | int]) -> None:
    root = Path(__file__).resolve().parent
    runfile = root / "Main_SVMDP.py"

    cmd = [sys.executable, str(runfile), *(str(arg) for arg in args)]
    subprocess.run(cmd, check=True, cwd=root)

# Every benchmark of the experiment: the model-specific arguments, and whether the dense (no RL) baseline is run
BENCHMARKS = {
    'MountainCar':      {'args': ["--model", "MountainCar"], 'dense': True},
    'Pendulum':         {'args': ["--model", "Pendulum"], 'dense': True},
    'CartPole':         {'args': ["--model", "CartPole"], 'dense': False},
    'CartPole_hard':    {'args': ["--model", "CartPole_hard"], 'dense': False},
    'Dubins3D':         {'args': ["--model", "Dubins3D"], 'dense': True},
    'Dubins4D':         {'args': ["--model", "Dubins4D"], 'dense': True},
    'Drone4D':          {'args': ["--model", "Drone4D"], 'dense': True},
    'Drone4D_damping':  {'args': ["--model", "Drone4D", "--damping", 0.1], 'dense': True},
    'Drone4D_2agent':   {'args': ["--model", "Drone4D_2agent"], 'dense': False},
    'Drone6D':          {'args': ["--model", "Drone6D"], 'dense': True},
    'Drone6D_small':    {'args': ["--model", "Drone6D_small"], 'dense': False},
    'Drone4D_battery':  {'args': ["--model", "Drone4D_battery", "--damping", 0.1], 'dense': True},
    'Drone6D_battery':  {'args': ["--model", "Drone6D_battery", "--damping", 0], 'dense': False},
}

# Arguments shared by all runs
COMMON_ARGS = ["--solver", "jax", "--satprob", 0.99]

# To run particular benchmarks, simply change the selection below
RUN_BENCHMARKS = ['Drone4D_2agent']
RUN_DENSE = False           # Also run the dense baseline (once, seed 0) for benchmarks that have one
ALGOS = ['sac']             # Any of 'ppo', 'sac'
SEEDS = [0]                 # E.g. range(5)
EXTRA_ARGS = []             # Appended to every run, e.g. ["--total_timesteps", 1000]

if __name__ == "__main__":

    queue = []
    for name in RUN_BENCHMARKS:
        benchmark = BENCHMARKS[name]
        if RUN_DENSE and benchmark['dense']:
            queue.append((f'{name} (dense)', [*benchmark['args'], *COMMON_ARGS, "--seed", 0, "--dense", *EXTRA_ARGS]))
        for algo in ALGOS:
            for seed in SEEDS:
                queue.append((f'{name} ({algo}, seed {seed})',
                              [*benchmark['args'], *COMMON_ARGS, "--seed", seed, "--algo", algo, *EXTRA_ARGS]))

    num_runs = len(queue)
    while queue:
        label, args = queue.pop(0)
        print(f'\n=== Run {num_runs - len(queue)}/{num_runs}: {label} ===', flush=True)
        run_fixed_SVMDP(args=args)
