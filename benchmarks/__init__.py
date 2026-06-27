# Load the benchmarks from the subfiles
from .Dubins3D import Dubins3D
from .Dubins4D import Dubins4D
from .Drone4D import Drone4D
from .Drone6D import Drone6D, Drone6D_small, Drone6D_medium
from .Pendulum import Pendulum
from .MountainCar import MountainCar
from .Integrators import DoubleIntegrator
from .Test1D import Test1D


def create_model(args):
	from core.abstraction.model import parse_linear_model, parse_nonlinear_model

	model_map = {
		'Dubins3D': Dubins3D,
		'Dubins4D': Dubins4D,
		'Drone4D': Drone4D,
		'Drone6D': Drone6D,
		'Drone6D_small': Drone6D_small,
		'Drone6D_medium': Drone6D_medium,
		'Pendulum': Pendulum,
		'MountainCar': MountainCar,
		'DoubleIntegrator': DoubleIntegrator,
		'Test1D': Test1D,
	}

	model_cls = model_map.get(args.model)
	if model_cls is None:
		raise ValueError(f"The passed model '{args.model}' could not be found")

	base_model = model_cls(args)
	if base_model.linear:
		return parse_linear_model(base_model)
	return parse_nonlinear_model(base_model)