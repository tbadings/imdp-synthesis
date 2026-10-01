import itertools
import logging
from pathlib import Path
import time
import numpy as np

from .base import BaseRL
from .config import RLConfig, resolve_rl_config
from .env import BenchmarkEnv
from .ppo import PPO
from .sac import SAC
from .tube import build_tube, rollout_sweep_priority
from .plotting import plot_rl_trajectories_with_active_states

logger = logging.getLogger(__name__)


def get_rl_algo(algo: str, env, cfg) -> BaseRL:
    """Factory creating an RL algorithm instance for the given environment and config."""
    return SAC(env, cfg) if str(algo).lower().strip() == "sac" else PPO(env, cfg)


def find_active(model, args, return_sweep_priority=False):
    """Find active states and discrete actions using reinforcement learning exploration.

    With return_sweep_priority, also return each active state's DP sweep priority (see
    rollout_sweep_priority), which is None unless --sweep_order is 'trajectory'.
    """
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
    goal_reached, newly_visited, trajectories = agent.evaluate(
        args,
        discrete_actions=discrete_actions,
        seed=args.seed,
        output_dir=out_dir,
        return_trajectories=True,
        return_visited_array=True,
    )
    logger.info("Goal reached in %d/%d evaluation episodes.", goal_reached, cfg.eval_episodes)

    # Tube construction (active states)
    t = time.time()
    active_states = build_tube(newly_visited, cfg, model, env, agent=agent, discrete_actions=discrete_actions)
    print('(Time to extract state tube: %.2f seconds)' % (time.time() - t))

    if args.plot_SA_tube:
        plot_dims = getattr(model, "plot_dimensions", None)
        if plot_dims is not None and len(plot_dims) == 2:
            plot_rl_trajectories_with_active_states(
                model,
                env,
                trajectories,
                active_states,
                list(plot_dims),
                out_dir,
                algo_name=cfg.rl_algo,
            )

    # Discretized active policy actions
    t = time.time()
    selected_actions = agent.get_policy_actions(
        active_states, discrete_actions, num=cfg.RL_actions_per_state
    )
    active_actions = {
        tuple(cell): selected_actions[i]
        for i, cell in enumerate(active_states.tolist())
    }
    print('(Time to extract active actions: %.2f seconds)' % (time.time() - t))

    if not return_sweep_priority:
        return active_states, active_actions, agent

    sweep_priority = None
    if getattr(args, "sweep_order", "random") == "trajectory":
        t = time.time()
        sweep_priority = rollout_sweep_priority(trajectories, active_states, env)
        if sweep_priority is None:
            logger.warning("No evaluation rollout reached the goal; keeping the random sweep order.")
        print('(Time to compute sweep priority: %.2f seconds)' % (time.time() - t))
    return active_states, active_actions, agent, sweep_priority


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
