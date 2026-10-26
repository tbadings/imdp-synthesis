# IMDP Synthesis

This repository implements abstraction-based methods for robust policy synthesis in discrete-time stochastic dynamical systems. The goal is to compute a policy that maximizes the probability of reaching a goal set while avoiding unsafe (critical) regions, with a formal lower bound on that probability.

The continuous system is abstracted into a finite model, on which robust dynamic programming computes a policy and a lower bound on the satisfaction probability. Two abstractions are implemented, each with its own entrypoint:

- **IMDP** (`Main_IMDP.py`): an interval Markov decision process.
- **SVMDP** (`Main_SVMDP.py`): a set-valued Markov decision process.

By default, the abstraction is restricted to a tube of states and actions around rollouts of a reinforcement learning (RL) policy (PPO or SAC). Use `--dense` to abstract the full state space instead.

> **Note:** At present, the SVMDP path performs much better than the IMDP path, and it is the recommended default. All experiment scripts in this repository (`ICLR_*.py`) use `Main_SVMDP.py`.

The codebase implements and builds on methods from the following papers:

- Thom Badings and Alessandro Abate. "Probabilistic Alternating Simulations for Policy Synthesis in Uncertain Stochastic Dynamical Systems." In 2025 IEEE 64th Conference on Decision and Control (CDC), pages 3919-3924. IEEE, 2025.
- Thom Badings, Licio Romao, Alessandro Abate, David Parker, Hasan A. Poonawala, Marielle Stoelinga, and Nils Jansen. "Robust Control for Dynamical Systems with Non-Gaussian Noise via Formal Abstractions." Journal of Artificial Intelligence Research, 76:341-391, 2023.

# Installation

## 1. Python environment

The project is documented and tested with Python 3.13.

We recommend creating a dedicated environment with conda or mamba:

```bash
conda create -n dynabs-jax python=3.13
conda activate dynabs-jax
```

Next, install the base Python dependencies (including Flax and Optax for the RL front-end, and PyVista/VTK for 3D rendering):

```bash
pip install -r requirements.txt
```

## 2. JAX installation

JAX should be installed differently depending on whether you wish to run the code on CPU or GPU.

For CPU-only execution:

```bash
pip install jax
```

For CUDA-enabled execution via conda:

```bash
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```

Apple Silicon note: in practice, CPU JAX has generally been more reliable here than JAX Metal.

## 3. Storm backend (optional, IMDP only)

The `storm` solver backend of the IMDP path requires the Storm model checker together with its Python bindings `stormpy`. If `stormpy` is not installed, the JAX backend remains fully available. The SVMDP path does not use Storm.

For installation instructions, see:

- Storm build documentation: https://www.stormchecker.org/documentation/obtain-storm/build.html
- Stormpy installation guide: https://stormchecker.github.io/stormpy/installation.html

# Pipeline overview

Both entrypoints share the same pipeline and command-line options (see `core/options.py`):

1. **Model.** Load the benchmark from `benchmarks/`.
2. **Partition.**
   - *Default (sparse):* train an RL policy (PPO or SAC), roll it out, and build a tube of grid cells around the rollouts. Each cell in the tube keeps the `RL_actions_per_state` discrete actions that the RL policy prefers most. The result is a `SparsePartition`.
   - *`--dense`:* partition the full state space into a grid with every discrete action enabled (`DensePartition`). RL is skipped.
3. **Forward reachability.** For every cell and action, compute the forward reachable set under each cell of the noise partition.
4. **Abstraction.** Build the IMDP or the SVMDP (see below).
5. **Synthesis.** Run robust value iteration or robust policy iteration (the default). With `--satprob p < 1`, synthesis stops as soon as the value in the initial state reaches `p`.
6. **Validation.** Simulate 1000 closed-loop Monte Carlo traces of the continuous system and report the empirical satisfaction probability.
7. **Output.** Save logs and plots, plus a checkpoint if `--save_checkpoint` is set.

## IMDP path (`Main_IMDP.py`)

The IMDP has one abstract state per partition cell. For each state-action pair, the code computes a probability interval for every successor cell, plus one for the absorbing (out-of-domain) state (`core/abstraction/imdp/probability_intervals.py`). Robust dynamic programming then optimizes against the worst-case distribution within these intervals.

- Solvers: `--solver jax` (robust value iteration or robust policy iteration in JAX) or `--solver storm` (robust value iteration via Storm).
- IMDP-specific options: `--batch_size`, `--decimals`, `--pAbs_min`.

```bash
python Main_IMDP.py --model MountainCar --solver jax --noise_distr gaussian
python Main_IMDP.py --model Test1D --solver storm --no-policy_iteration
```

Note: the Storm backend supports only robust value iteration. If you select `--solver storm` with policy iteration (the default), the code emits a warning and falls back to robust value iteration.

## SVMDP path (`Main_SVMDP.py`)

The SVMDP also has one abstract state per partition cell, but its transitions are set-valued. For each state-action pair and each noise cell, the system moves, with that cell's exact probability, to some cell within a box of successor cells. The choice of cell within that box is adversarial (nondeterministic). The robust value of a state-action pair is therefore the probability-weighted sum, over the noise cells, of the minimum value over each successor box.

The SVMDP never computes probability intervals for individual successor cells:

- **Forward reachability** (`core/abstraction/svmdp/forward_reachability.py`) stores, for each state-action pair and each dimension, the merged grid-index intervals of the successors, together with the probability of each successor box.
- **Dynamic programming** (`core/abstraction/svmdp/dynprog.py`) keeps the values on a tiled grid (`successor_ids.py`) and reduces one dimension at a time, so the successor boxes are never formed explicitly.

Differences from the IMDP path:

- It always uses its own JAX dynamic programming, so `--solver` is ignored.
- `--sweep_order {trajectory,random}` sets the order of the state sweeps. The default `trajectory` sweeps states nearest the goal along the RL rollouts first. Dense runs always use a random order.
- `--no-mc_simulations` skips Monte Carlo validation, for example in parameter sweeps.
- It writes a `summary.csv` with timings, abstraction size, the optimal value, and the empirical satisfaction probability.

```bash
python Main_SVMDP.py --model Drone4D --algo ppo --satprob 0.99
python Main_SVMDP.py --model MountainCar --dense
```

# Available benchmarks

| Name | State dim | Input dim | Description |
| --- | --- | --- | --- |
| `Test1D` | 1 | 1 | Simple 1D test model. |
| `DoubleIntegrator` | 2 | 1 | Double integrator. |
| `Pendulum` | 2 | 1 | Inverted pendulum. |
| `MountainCar` | 2 | 1 | Mountain car. |
| `Dubins3D` | 3 | 2 | Dubins vehicle. |
| `Dubins4D` | 4 | 2 | Dubins vehicle with velocity state. |
| `CartPole` | 4 | 1 | Cart-pole balancing. |
| `CartPole_hard` | 4 | 1 | CartPole with an obstacle hanging above the middle of the track, so the pole must tilt to pass underneath. |
| `Drone4D` | 4 | 2 | Drone with 2D position and velocity. |
| `Drone4D_battery` | 5 | 2 | Drone4D with a battery state. |
| `Drone6D` | 6 | 3 | Drone with 3D position and velocity. |
| `Drone6D_small` | 6 | 3 | Reduced Drone6D configuration for faster debugging. |
| `Drone6D_battery` | 7 | 3 | Drone6D with a battery state. |
| `Drone4D_2agent` | 8 | 4 | Two independent Drone4D agents. |

Each benchmark defines its default partition, action grid, noise model, and RL settings (`rl_config`, an instance of `core.rl.config.RLConfig`). The RL options on the command line override the benchmark's `rl_config`.

# Command-line options

Unless marked otherwise, options apply to both entrypoints. Run `python Main_SVMDP.py --help` for the full list.

## Model and abstraction

- `--model`: benchmark name (see above).
- `--noise_distr {gaussian,normal,triangular}`: noise model. `normal` is an alias of `gaussian`.
- `--damping`: velocity damping for the drone benchmarks.
- `--model_version`: model variant (only used by some benchmarks).
- `--dense`: use a dense partition and skip RL.
- `--fix_num_states`, `--fix_num_actions`, `--fix_num_choices`: rescale the benchmark's state or action grid to approximately fit a budget. These require `--dense`.
- `--shrink_frs`: shrink forward reachable sets slightly inward for numerical stability.

## Synthesis

- `--policy_iteration` / `--no-policy_iteration`: robust policy iteration (default) or robust value iteration.
- `--satprob`: stop synthesis once the value in the initial state reaches this bound (default `1`, i.e. run to convergence).
- `--solver {jax,storm}`: *IMDP only.*
- `--sweep_order {trajectory,random}`: *SVMDP only.*

## Reinforcement learning (sparse partition)

- `--algo {ppo,sac}` (alias `--rl_algo`): RL algorithm.
- `--load_policy PATH`: load a saved RL policy (`rl_policy.pkl`) instead of training.
- `--total_timesteps`, `--eval_episodes`, `--max_steps`, `--eval_steps`: training and rollout budgets.
- `--RL_actions_per_state`: number of discrete actions kept per state in the tube.
- `--tube_method {inflation,smart}`, `--inflation_rate`, `--smart_tube_rate`: how the tube around the rollouts is constructed.
- Reward shaping (`--goal_reward`, `--unsafe_penalty`, `--distance_cost`, ...) and PPO/SAC hyperparameters: see `--help` and `core/rl/config.py`.

## Performance and memory tuning

- `--frs_batch_size`: number of regions processed per batch when computing forward reachable sets.
- `--batch_size`: *IMDP only.* Number of states processed together when computing transition probability intervals. If you run into memory limits on the IMDP path, reduce this first. On larger benchmarks, values such as `1000`, `100`, or even smaller may be necessary.
- `--gpu` / `--no-gpu`: select the main JAX platform.
- `--gpu_rvi` / `--no-gpu_rvi`: select the device used for robust dynamic programming.

## Reproducibility, output, and plotting

- `--seed`: random seed for NumPy, JAX, and RL training.
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: logging verbosity.
- `--output_root`: root directory for timestamped run folders (default `output`).
- `--save_checkpoint`, `--load_checkpoint PATH`: see below.
- `--mc_simulations` / `--no-mc_simulations`: *SVMDP only.* Run or skip Monte Carlo validation.
- `--plot_title`, `--plot_grid`, `--plot_ticks` (each with a `--no-` variant).
- `--plot_SA_tube`: plot the state-action tube around the RL rollouts (2D plot dimensions only).
- `--paper_figures`: use the figure style from the paper.

# Reusing a checkpoint

With `--save_checkpoint`, a checkpoint is saved after policy synthesis and Monte Carlo validation, before plotting. To regenerate plots without rerunning abstraction, synthesis, or simulations, load it with the same entrypoint that created it:

```bash
python Main_SVMDP.py --load_checkpoint output/<timestamp>_<model>/checkpoint.pkl
python Main_IMDP.py --load_checkpoint output/<timestamp>_<model>/checkpoint.pkl
```

The checkpoint includes the model, partition, abstraction, synthesized values and policy, and simulation traces. SVMDP checkpoints also include the run summary. Current plotting options (such as `--plot_title`) apply to the regenerated figures, which are written to a new output directory. Checkpoints must contain completed synthesis and simulation results; abstraction-only checkpoints are not supported.

To reuse only the trained RL policy and rebuild everything else, pass `--load_policy output/<timestamp>_<model>/rl_policy.pkl`.

# Output structure

Each run creates an output directory named like:

```text
output/2026-06-02_10-49-08_MountainCar/
```

Depending on the entrypoint, benchmark, and options, that directory may contain:

- `run_<timestamp>.log`: full log output.
- `rl_policy.pkl`: the trained RL policy (sparse runs).
- `rl_trajectories.{png,pdf}`: RL evaluation rollouts. With `--plot_SA_tube`, also `rl_trajectories_active_states.{png,pdf}`.
- `summary.csv` (SVMDP only): RL, abstraction, and synthesis times; number of states, actions, and state-action pairs; optimal value in the initial state; empirical satisfaction probability.
- `checkpoint.pkl` (with `--save_checkpoint`).
- Heatmaps of the synthesized values, and trajectory plots of the Monte Carlo traces.
- Benchmark-specific animations: GIFs for Pendulum, MountainCar, and CartPole. With SVMDP, also a trace plot and GIF for Drone4D_2agent.

## Drone6D 3D rendering

Drone6D runs (including its variants) also export `traces_3d_pyvista_<timestamp>.png`, alongside a Matplotlib 3D plot (PNG/PDF) as a backup. The PyVista renderer lives in `core/plotting/drone3d.py`. It uses a view at 60° elevation and 25° azimuth, with orthographic framing fitted to the domain. The surface tiles are decorative, with a spacing of up to two world units, and do not represent abstraction cells. Paths use a shape-preserving cubic spline through the simulation samples.

Rendering runs in a separate process, so a native OpenGL crash does not interrupt the benchmark. Failures are logged, and the Matplotlib backup remains available.

# Experiment scripts

The `ICLR_*.py` launchers run batches of `Main_SVMDP.py` with fixed arguments:

- `ICLR_Q1.py`: main benchmark runs. Edit `RUN_BENCHMARKS`, `ALGOS`, `SEEDS`, and `RUN_DENSE` at the top of the file to choose what to run.
- `ICLR_Q1-fixed-buget.py`: dense baselines under fixed choice budgets (`--dense --fix_num_choices ...`).
- `ICLR_Q2_a.py`, `ICLR_Q2_b.py`: sweeps over `--inflation_rate` and `--RL_actions_per_state` for Dubins3D and Drone4D.
- `ICLR_figures.py`: runs that produce the paper figures.

`experiment_logs/experiment_fixed_budget/` contains the summaries from the fixed-budget experiment and `create_table.py`, which turns them into a LaTeX table.

# Repository structure

- `Main_IMDP.py`, `Main_SVMDP.py`: entrypoints for the two abstraction paths.
- `benchmarks/`: benchmark definitions (`models.py` holds the dynamics; `dynamics/` holds the noise distributions and interval arithmetic).
- `core/options.py`: command-line options shared by both entrypoints.
- `core/rl/`: RL front-end (PPO, SAC, environment wrapper, tube construction).
- `core/abstraction/`: model parsing, partitions (`partition.py`), and the two abstractions:
  - `imdp/`: forward reachability, probability intervals, and the JAX and Storm solvers.
  - `svmdp/`: forward reachability, the tiled value grid, and the dynamic programming solver.
- `core/validate/`: Monte Carlo simulation.
- `core/plotting/`: heatmaps, trajectory plots, and the PyVista Drone6D renderer.
- `experiment_logs/`: stored experiment results.
