# IMDP Synthesis

This repository implements abstraction-based methods for robust policy synthesis in discrete-time stochastic dynamical systems, based on interval Markov decision process (IDMP) abstractions.

In particular, the codebase implements and builds on methods from the following papers:

- Thom Badings and Alessandro Abate. "Probabilistic Alternating Simulations for Policy Synthesis in Uncertain Stochastic Dynamical Systems." In 2025 IEEE 64th Conference on Decision and Control (CDC), pages 3919-3924. IEEE, 2025.
- Thom Badings, Licio Romao, Alessandro Abate, David Parker, Hasan A. Poonawala, Marielle Stoelinga, and Nils Jansen. "Robust Control for Dynamical Systems with Non-Gaussian Noise via Formal Abstractions." Journal of Artificial Intelligence Research, 76:341-391, 2023.

# Installation

Follow the steps below to install the tool:

## 1. Python environment

The project is currently documented and tested with Python 3.13.

We recommend creating a dedicated environment with conda or mamba:

```bash
conda create -n dynabs-jax python=3.13
conda activate dynabs-jax
```

Next, install the base Python dependencies:

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

## 3. Storm backend (optional)

The `storm` solver backend requires the Storm model checker together with the Python bindings `stormpy`. If `stormpy` is not installed, the JAX backend remains fully available.

For installation instructions, see:

- Storm build documentation: https://www.stormchecker.org/documentation/obtain-storm/build.html
- Stormpy installation guide: https://stormchecker.github.io/stormpy/installation.html

# Project entrypoints

The main script is:

```bash
python Main_IMDP.py --model <model_name>
```

The repository also includes a convenience launcher, `Run_fixed_benchmark.py`, for replaying a fixed configuration during local experimentation. It is not intended as the primary public entrypoint and currently uses hardcoded parameters.

# Available benchmarks

The current benchmark registry includes:

- `Dubins3D`: 3D Dubins vehicle with 2D control input.
- `Dubins4D`: 4D Dubins vehicle with 2D control input.
- `Drone4D`: 4D quadrotor model.
- `Drone6D`: 6D quadrotor model.
- `Drone6D_small`: reduced 6D quadrotor configuration for faster debugging.
- `Pendulum`: inverted pendulum benchmark.
- `MountainCar`: mountain car benchmark.
- `DoubleIntegrator`: double integrator benchmark.
- `Test1D`: simple 1D test model.

Example runs:

```bash
python Main_IMDP.py --model Dubins3D
python Main_IMDP.py --model Drone4D
python Main_IMDP.py --model MountainCar
```

# Running a benchmark

A typical command looks as follows:

```bash
python Main_IMDP.py --model MountainCar --solver jax --noise_distr gaussian
```

This will:

1. create a timestamped output directory under `output/`,
2. construct the partition and IMDP abstraction,
3. solve the robust reachability problem,
4. run Monte Carlo validation,
5. save logs, plots, and a `checkpoint.pkl` file.

## Reusing a checkpoint

To skip abstraction generation and rerun the solver and plotting pipeline from a saved checkpoint:

```bash
python Main_IMDP.py --model MountainCar --load_checkpoint output/<timestamp>_MountainCar/checkpoint.pkl
```

When `--load_checkpoint` is provided, the saved model, partition, and IMDP are loaded directly from the specified pickle file.

# Solver backends

Two solver backends are supported:

- `--solver jax`: robust dynamic programming implemented in JAX.
- `--solver storm`: robust value iteration through Storm.

Examples:

```bash
python Main_IMDP.py --model Pendulum --solver jax
python Main_IMDP.py --model Test1D --solver storm --no-policy_iteration
```

Note: the Storm backend currently supports robust value iteration, not robust policy iteration. If `--solver storm` is selected together with policy iteration, the code emits a warning and falls back to robust value iteration.

# Important command-line options

## Model and solver selection

- `--model`: benchmark name.
- `--solver {jax,storm}`: select the backend.
- `--noise_distr {gaussian,normal,triangular}`: select the noise model. `normal` is accepted as an alias of `gaussian`.

## Performance and memory tuning

- `--batch_size`: number of states processed together when computing transition probability intervals.
- `--frs_batch_size`: number of regions processed per batch when computing forward reachable sets.
- `--gpu` or `--no-gpu`: select the main JAX platform.
- `--gpu_rvi` or `--no-gpu_rvi`: select the device used for robust dynamic programming.

If you run into memory limits, reduce `--batch_size` first. On larger benchmarks, values such as `1000`, `100`, or even smaller may be necessary.

## Reproducibility and logging

- `--seed`: random seed for NumPy and JAX.
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: logging verbosity.
- `--output_root`: root directory for timestamped run folders.

## Plotting controls

- `--plot_title` or `--no-plot_title`
- `--plot_grid` or `--no-plot_grid`
- `--plot_ticks` or `--no-plot_ticks`

# Output structure

Each run creates an output directory named like:

```text
output/2026-06-02_10-49-08_MountainCar/
```

Depending on the benchmark and selected options, that directory may contain:

- `checkpoint.pkl`: serialized model, partition, IMDP, and selected arguments.
- `run_<timestamp>.log`: full log output.
- heatmaps of values and selected inputs.
- trajectory plots.
- benchmark-specific GIFs for models such as Pendulum or MountainCar.

# Running tests

The repository includes a small regression test suite under `tests/`.

Run all tests with:

```bash
python tests/run_tests.py
```

List available test modules:

```bash
python tests/run_tests.py --list
```

Run specific tests:

```bash
python tests/run_tests.py test_options
python tests/run_tests.py test_benchmarks test_test1d_probability_intervals
```

# Repository structure

- `benchmarks/`: benchmark model definitions.
- `core/abstraction/`: partitioning, IMDP construction, and solver implementations.
- `core/plotting/`: heatmaps and trajectory visualization.
- `core/validate/`: Monte Carlo simulation and validation utilities.
- `tests/`: regression tests.