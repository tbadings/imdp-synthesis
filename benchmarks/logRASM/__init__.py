# Load the benchmarks from the subfiles
from .collision_avoidance import CollisionAvoidance
from .drone4D import Drone4D
from .planar_robot import PlanarRobot
from .triple_integrator import TripleIntegrator


def get_model_fun(model_name):
    if model_name == 'CollisionAvoidance':
        envfun = CollisionAvoidance
    elif model_name == 'TripleIntegrator':
        envfun = TripleIntegrator
    elif model_name == 'PlanarRobot':
        envfun = PlanarRobot
    elif model_name == 'Drone4D':
        envfun = Drone4D
    else:
        assert False, f"Unknown model name: {model_name}"

    return envfun
