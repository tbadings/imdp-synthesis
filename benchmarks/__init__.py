# Load the benchmarks from the subfiles
from math import prod

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
	'''
	Scale each grid dimension proportionally and round to the nearest integer.
	'''
	
	if budget <= 0:
		raise ValueError("Partition budgets must be positive.")

	defaults = np.array(default_counts, dtype=int)
	if any(count <= 0 for count in defaults):
		raise ValueError("Partition and action grid sizes must be positive.")
	if budget < 1:
		return [1] * len(defaults)

	scale = (budget / prod(defaults)) ** (1 / len(defaults))
	counts = [max(1, round(count * scale)) for count in defaults]

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

	states_budget = getattr(args, 'fix_num_states', None)
	actions_budget = getattr(args, 'fix_num_actions', None)
	choices_budget = getattr(args, 'fix_num_choices', None)

	if actions_budget is not None:
		base_model.num_actions = np.array(
			_fit_grid_budget(base_model.num_actions, actions_budget), dtype=int
		)

	if choices_budget is not None:
		num_actions = np.prod(base_model.num_actions)
		base_model.partition['number_per_dim'] = np.array(
			_fit_grid_budget(state_counts, choices_budget / num_actions), dtype=int
		)

	elif states_budget is not None:
		base_model.partition['number_per_dim'] = np.array(
			_fit_grid_budget(state_counts, states_budget), dtype=int
		)

	if base_model.linear:
		return parse_linear_model(base_model)
	return parse_nonlinear_model(base_model)
