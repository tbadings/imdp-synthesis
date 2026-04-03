'''
This is the main Python file for DynAbs-JAX.
The file can be run from the terminal as

```Python3 RunFile.py --model <model-name> ...```

For all available arguments, please see the function :func:`core.options.parse_arguments`.
'''

import datetime
import os
import time
from pathlib import Path
import jax
import numpy as np
import sys

import benchmarks
from core.Gaussian_probabilities import compute_probability_intervals
from core.forward_reachability import RectangularForward
from core.model import parse_linear_model, parse_nonlinear_model
from core.options import parse_arguments
from core.partition import RectangularPartition
from core.imdp import IMDP

# Uncomment one of the following lines to run an example benchmark.
# If it seems to be 'stuck' when computing the transition probabilities, consider decreasing the batch size (e.g., to 1000).
sys.argv = ['RunFile.py', '--model', 'Dubins3D', '--batch_size', '1000']
# sys.argv = ['RunFile.py', '--model', 'Dubins4D', '--batch_size', '1000']
# sys.argv = ['RunFile.py', '--model', 'Pendulum', '--batch_size', '1000']
# sys.argv = ['RunFile.py', '--model', 'MountainCar', '--batch_size', '1000', '--plot_title']
# sys.argv = ['RunFile.py', '--model', 'DoubleIntegrator', '--batch_size', '30000', '--plot_title']
# sys.argv = ['RunFile.py', '--model', 'Drone3D_small', '--batch_size', '100', '--plot_title']
# sys.argv = ['RunFile.py', '--model', 'Drone3D', '--batch_size', '10000', '--plot_title']
# sys.argv = ['RunFile.py', '--model', 'Drone2D', '--batch_size', '1000', '--plot_title']

if __name__ == '__main__':
    jax.config.update("jax_default_matmul_precision", "high")

    args = parse_arguments()
    args.floatprecision = np.float32
    if args.gpu:
        jax.config.update('jax_platform_name', 'gpu')
        print('- Requested to run on GPU')
    else:
        jax.config.update('jax_platform_name', 'cpu')
        print('- Requested to run on CPU')
    if args.gpu_rvi:
        args.rvi_device = jax.devices('gpu')[0]
        print('- Requested to run RVI on GPU')
    else:
        args.rvi_device = jax.devices('cpu')[0]
        print('- Requested to run RVI on CPU')

    print('=== JAX STATUS ===')
    print(f'Devices available: {jax.devices()}')
    from jax.extend.backend import get_backend

    print(f'Jax runs on: {get_backend().platform}')
    print('==================\n')

    np.random.seed(args.seed)
    args.jax_key = jax.random.PRNGKey(args.seed)

    # In debug mode, configure jax to use Float64 (for more accurate computations)
    if args.debug:
        from jax import config

        config.update("jax_enable_x64", True)

    # Set current working directory
    args.cwd = os.path.dirname(os.path.abspath(__file__))
    args.root_dir = Path(args.cwd)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f'Run started at {stamp} using arguments:')
    for key, val in vars(args).items():
        print(' - `' + str(key) + '`: ' + str(val))
    print('\n==============================\n')

    # Define and parse model
    if args.model == 'Dubins3D':
        base_model = benchmarks.Dubins3D(args)
    elif args.model == 'Dubins4D':
        base_model = benchmarks.Dubins4D(args)
    elif args.model == 'Drone2D':
        base_model = benchmarks.Drone2D(args)
    elif args.model == 'Drone3D':
        base_model = benchmarks.Drone3D(args)
    elif args.model == 'Drone3D_small':
        base_model = benchmarks.Drone3D_small(args)
    elif args.model == 'Pendulum':
        base_model = benchmarks.Pendulum(args)
    elif args.model == 'MountainCar':
        base_model = benchmarks.MountainCar(args)
    elif args.model == 'DoubleIntegrator':
        base_model = benchmarks.DoubleIntegrator(args)
    else:
        assert False, f"The passed model '{args.model}' could not be found"

    t = time.time()

    # Parse given model
    if base_model.linear:
        model = parse_linear_model(base_model)
    else:
        model = parse_nonlinear_model(base_model)

    # Create partition of the continuous state space into convex polytope
    partition = RectangularPartition(model=model)
    
    # Create actions based on forward reachable sets
    actions = RectangularForward(args=args, partition=partition, model=model)
    actions_inputs = actions.id_to_input
    
    P_full, S_id, A_id, P_absorbing = compute_probability_intervals(args=args, 
                                                                    model=model, 
                                                                    partition=partition, 
                                                                    actions=actions,
                                                                    vectorized=True)

    # assert False
    del actions

    imdp = IMDP(partition=partition,
                states=np.array(partition.regions['idxs']),
                actions_inputs=actions_inputs,
                x0=model.x0,
                goal_regions=np.array(partition.goal['bools']),
                critical_regions=np.array(partition.critical['bools']),
                P_full=P_full,
                S_id=S_id,
                A_id=A_id,
                P_absorbing=P_absorbing)

    print(f'- Generating abstraction took: {(time.time() - t):.3f} sec.')

    # %% Build and verify with JAX-based RVI

    from core.imdp import RVI_JAX

    print('Compute optimal policy via robust value iteration with JAX...')

    with jax.default_device(args.rvi_device):

        t = time.time()
        V, policy, policy_inputs = RVI_JAX(
            args=args, 
            imdp=imdp, 
            s0=partition.x2state(model.x0)[0], 
            max_iterations=10000, 
            epsilon=1e-6, 
            RND_SWEEPS=True, 
            BATCH_SIZE=1000, 
            policy_iteration=args.policy_iteration)
        print (f'- RVI with JAX (random-batched asynchronous) took: {(time.time() - t):.3f} sec.')

    # %% Simulations and plot

    sim_policy = policy
    sim_policy_inputs = policy_inputs
    sim_values = V

    from core.simulate import MonteCarloSim
    from plotting.traces import plot_traces
    from plotting.heatmap import heatmap

    sim = MonteCarloSim(model, partition, sim_policy, sim_policy_inputs, model.x0, verbose=False, iterations=1000)
    print('Empirical satisfaction probability:', sim.results['satprob'])

    plot_traces(args, stamp, model.plot_dimensions, partition, model, sim.results['traces'], line=False, num_traces=10, add_unsafe_box=False,)
    heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=sim_values, filename="heatmap_satprob")
    heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=sim_policy_inputs[:,0], filename="heatmap_inputs")
    
    if args.model == 'Pendulum':
        model.plot_trajectory_gif(np.array(sim.results['traces'][0]['x'])[:,0], filename=f'output/pendulum_{stamp}.gif')

    if args.model == 'MountainCar':
        model.plot_trajectory_gif(np.array(sim.results['traces'][0]['x'])[:,0], filename=f'output/mountaincar_{stamp}.gif')
        
# %%
