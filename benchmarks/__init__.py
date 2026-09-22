# Load the benchmarks from the subfiles
from math import floor, prod

import numpy as np

from .Dubins3D import Dubins3D
from .Dubins4D import Dubins4D
from .Drone4D import Drone4D, Drone4D_battery
from .Drone6D import Drone6D, Drone6D_small, Drone6D_battery
from .Pendulum import Pendulum
from .MountainCar import MountainCar
from .CartPole import CartPole
from .Integrators import DoubleIntegrator
from .Test1D import Test1D


def _fit_grid_budget(default_counts, budget):
	"""Scale a Cartesian grid to fit a budget while retaining its default proportions."""
	if budget <= 0:
		raise ValueError("Partition budgets must be positive integers.")

	defaults = [int(count) for count in default_counts]
	if any(count <= 0 for count in defaults):
		raise ValueError("Partition and action grid sizes must be positive.")
	if budget == prod(defaults):
		return defaults

	scale = (budget / prod(defaults)) ** (1 / len(defaults))
	counts = [max(1, floor(count * scale)) for count in defaults]
	current = prod(counts)
	# Clamping small dimensions to one can make the initial grid exceed the budget.
	while current > budget:
		i = max((i for i, count in enumerate(counts) if count > 1),
		        key=lambda i: counts[i] / defaults[i])
		current = current // counts[i] * (counts[i] - 1)
		counts[i] -= 1

	# Spend any remaining budget on the most undersampled dimension.
	while True:
		feasible = [i for i, count in enumerate(counts)
		            if current // count * (count + 1) <= budget]
		if not feasible:
			break
		i = min(feasible, key=lambda i: (counts[i] / defaults[i], i))
		current = current // counts[i] * (counts[i] + 1)
		counts[i] += 1

	return counts


def create_model(args):
	from core.abstraction.model import parse_linear_model, parse_nonlinear_model

	model_map = {
		'Dubins3D': Dubins3D,
		'Dubins4D': Dubins4D,
		'Drone4D': Drone4D,
		'Drone4D_battery': Drone4D_battery,
		'Drone6D': Drone6D,
		'Drone6D_small': Drone6D_small,
		'Drone6D_battery': Drone6D_battery,
		'Pendulum': Pendulum,
		'MountainCar': MountainCar,
		'CartPole': CartPole,
		'DoubleIntegrator': DoubleIntegrator,
		'Test1D': Test1D,
	}

	model_cls = model_map.get(args.model)
	if model_cls is None:
		raise ValueError(f"The passed model '{args.model}' could not be found")

	base_model = model_cls(args)
	state_counts = base_model.partition['number_per_dim']
	choice_budget = getattr(args, 'partition_choices', None)
	state_budget = getattr(args, 'partition_states', None)

	if choice_budget is not None:
		# A choice is a state-action pair. Scale both grids using one budget.
		num_state_dims = len(state_counts)
		counts = _fit_grid_budget([*state_counts, *base_model.num_actions], choice_budget)
		base_model.partition['number_per_dim'] = np.array(counts[:num_state_dims], dtype=int)
		base_model.num_actions = counts[num_state_dims:]
	elif state_budget is not None:
		base_model.partition['number_per_dim'] = np.array(
			_fit_grid_budget(state_counts, state_budget), dtype=int
		)

	if base_model.linear:
		return parse_linear_model(base_model)
	return parse_nonlinear_model(base_model)
