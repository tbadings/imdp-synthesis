import itertools
import logging
from pathlib import Path
import numpy as np

from .base import BaseRL
from .config import RLConfig, resolve_rl_config
from .env import BenchmarkEnv
from .ppo import PPO
from .sac import SAC
from .tube import build_tube

logger = logging.getLogger(__name__)


def get_rl_algo(algo: str, env, cfg) -> BaseRL:
    """Factory creating an RL algorithm instance for the given environment and config."""
    return SAC(env, cfg) if str(algo).lower().strip() == "sac" else PPO(env, cfg)


def find_active(model, args):
    """Find active states and discrete actions using reinforcement learning exploration."""
    cfg = resolve_rl_config(model, args)
    env = BenchmarkEnv(model, cfg)
    agent = get_rl_algo(cfg.rl_algo, env, cfg)

    out_dir = Path(getattr(args, "output_dir", "output"))

    # Load checkpoint or train
    if getattr(args, "load_policy", None):
        agent.load(args.load_policy)
    else:
        agent.train(seed=args.seed)
        agent.save(out_dir / "rl_policy.pkl")

    # Discretize continuous control space
    discrete_actions_per_dim = [
        np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i])
        for i in range(len(model.num_actions))
    ]
    discrete_actions = np.array(list(itertools.product(*discrete_actions_per_dim)), dtype=np.float32)

    # Rollouts and visited cell extraction
    goal_reached, newly_visited = agent.evaluate(discrete_actions=discrete_actions, seed=args.seed, output_dir=out_dir)
    logger.info("Goal reached in %d/%d evaluation episodes.", goal_reached, cfg.eval_episodes)

    # Tube construction (active states)
    active_states = build_tube(newly_visited, cfg, model, env, agent=agent, discrete_actions=discrete_actions)

    # Discretized active policy actions
    top_k = agent.get_policy_actions(active_states, discrete_actions, num=cfg.RL_actions_per_state)
    active_actions = {tuple(cell): top_k[i] for i, cell in enumerate(active_states.tolist())}

    return active_states, active_actions, agent


__all__ = [
    "find_active",
    "BenchmarkEnv",
    "RLConfig",
    "resolve_rl_config",
    "get_rl_algo",
    "BaseRL",
    "PPO",
    "SAC",
    "build_tube",
]
