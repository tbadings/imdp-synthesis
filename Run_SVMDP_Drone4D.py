import subprocess
import sys
from pathlib import Path

def run_fixed_SVMDP(args: list[str | int]) -> None:
    root = Path(__file__).resolve().parent
    runfile = root / "Main_SVMDP.py"

    cmd = [sys.executable, str(runfile), *(str(arg) for arg in args)]
    subprocess.run(cmd, check=True, cwd=root)

if __name__ == "__main__":
    root = Path(__file__).resolve().parent

    inflation_rates = [
        ["1", "1", "1", "1"],
        ["2", "1", "2", "1"], 
        ["4", "2", "4", "2"],
        ["6", "3", "6", "3"],
    ]
    active_actions = [1, 4, 9, 16, 25]

    for inf in inflation_rates:
        for act in active_actions:
            run_fixed_SVMDP(args=[
                "--model", "Drone4D",
                "--solver", "jax",
                "--seed", "0",
                "--algo", "ppo",
                "--inflation_rate", *inf,
                "--RL_actions_per_state", str(act),
                "--no-mc_simulations",
            ])
