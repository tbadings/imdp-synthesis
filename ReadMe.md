# Installation

The code can be installed by following the instructions below.

### 1. Create Python environment

The first step is to create the Python environment. We tested the code on Python 3.12.

We recommend using (mini)conda for setting up an environment. To create and activate the environment, run:

```
conda create -n dynabs-jax python=3.12
conda activate dynabs-jax
```

Install cddlib and GMP by [following the (OS-dependent) instructions here](https://pycddlib.readthedocs.io/en/latest/quickstart.html). For example, on MacOS, you can run:

```
brew install cddlib gmp
```

Then, install the dependencies within the conda environment:

```
pip install -r requirements.txt
```

Finally, install pycddlib:

```
pip install pycddlib
```

If installing pycddlib gives you an error similar to ```Cannot open include file: 'cddlib/setoper.h': No such file or directory```, then try
to [use this troubleshoot page.](https://pycddlib.readthedocs.io/en/latest/quickstart.html#installation)
On MacOS, the suggested fix is as follows:

```
env "CFLAGS=-I$(brew --prefix)/include -L$(brew --prefix)/lib" python -m pip install pycddlib
```

### 2. Install JAX

To install JAX with CUDA support via conda, run:

```
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```

To instead install JAX without CUDA support, run:

```
pip install jax==0.8.0
```

> We have also tested running with JAX on METAL. However, performance on Apple Silicon chips currently seems better on an up-to-date version of JAX+JAXlib (running on CPU) than on JAX on METAL.

# Running benchmarks

The following benchmarks are implemented and available to run:

- **Dubins3D**: 3D Dubins vehicle with 2D control input
- **Dubins4D**: 4D Dubins vehicle with 2D control input
- **Drone2D**: 2D quadrotor model
- **Drone3D**: 3D quadrotor model
- **Drone3D_small**: Smaller version of the 3D quadrotor (useful for faster debugging)
- **Pendulum**: Inverted pendulum system
- **MountainCar**: Mountain car benchmark
- **DoubleIntegrator**: Double integrator system
- **Test1D**: Simple 1D test model

To run a benchmark, use:

```
python RunFile.py --model <model_name>
```

For example:

```
python RunFile.py --model Dubins3D
python RunFile.py --model Drone2D
python RunFile.py --model MountainCar
```

Created figures will be stored in the `output/` folder. The runtimes and model sizes can be read from the terminal output.

## Noise distribution options

The tool supports different noise distribution types via the `--noise_distr` argument:

- **gaussian** (default): Gaussian (normal) distribution. A good choice for models with additive Gaussian noise.
- **triangular**: Symmetric triangular distribution. Useful for models with bounded, symmetric noise around a mean value.

Example usage:

```
python RunFile.py --model Dubins3D --noise_distr triangular
python RunFile.py --model Pendulum --noise_distr gaussian
```

## Additional options

- `--batch_size`: Number of states to process in a vectorized fashion when computing transition probability intervals. Default is 100. Increase for faster computation (but higher memory usage), or decrease if encountering memory issues. For example: `--batch_size 1000` or `--batch_size 10000`.
- `--policy_iteration`: Run policy iteration (default: True) or value iteration (False).
- `--gpu`: Run computations on GPU (requires CUDA-compatible hardware).
- `--seed`: Random seed for reproducibility (default: 0).
- `--plot_title`, `--plot_grid`, `--plot_ticks`: Toggle various plotting options.

## Memory troubleshooting

In case you run into memory issues, try decreasing the `--batch_size` argument. Good starting values are 30,000 or 10,000 for large models, and 1,000 for smaller models.

