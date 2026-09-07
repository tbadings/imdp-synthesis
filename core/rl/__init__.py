import itertools
import logging
from pathlib import Path
import pickle
from time import time
import jax
import numpy as np

from .config import resolve_rl_config
from .env import BenchmarkEnv
from .evaluation import evaluate_policy
from .policy import ActorCritic, find_policy_actions_batch
from .ppo import train_ppo
from .tube import _inflate_cells, _smart_inflate_cells

logger = logging.getLogger(__name__)

def find_active(model, args):
    cfg = resolve_rl_config(model, args)
    logger.info("Resolved RL config: %s", cfg)
    env = BenchmarkEnv(model, cfg)

    load_policy = args.load_policy
    if load_policy:
        p = Path(load_policy)
        p = p / "rl_policy.pkl" if p.is_dir() else (p if p.exists() else p.with_suffix(".pkl"))
        with open(p, "rb") as f:
            saved = pickle.load(f)
        params = saved["params"] if isinstance(saved, dict) and "params" in saved else saved
        actor_critic = ActorCritic(action_dim=len(env.u_min), pi_arch=tuple(cfg.pi_arch), vf_arch=tuple(cfg.vf_arch))
        logger.info("Loaded RL policy from %s; skipped training.", p)
    else:
        actor_critic, params = train_ppo(env=env, cfg=cfg, seed=args.seed)
        out_dir = Path(getattr(args, "output_dir", "output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "rl_policy.pkl", "wb") as f:
            pickle.dump({"params": jax.tree_util.tree_map(np.asarray, params)}, f)
        logger.info("Saved RL policy to %s", out_dir / "rl_policy.pkl")

    # Discretize continuous control space
    discrete_actions_per_dim = [
        np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i])
        for i in range(len(model.num_actions))
    ]
    discrete_actions = np.array(list(itertools.product(*discrete_actions_per_dim)), dtype=np.float32)

    # Policy evaluation rollouts
    goal_reached, newly_visited, _ = evaluate_policy(
        actor_critic=actor_critic, params=params,
        base_model=model, env=env, cfg=cfg,
        dims=list(model.plot_dimensions), output_dir=Path(getattr(args, "output_dir", "output")),
        discrete_actions=discrete_actions, seed=args.seed,
    )
    logger.info(f"Goal reached in {goal_reached}/{cfg.eval_episodes} episodes.")

    # Compute the tube (active states)
    number_per_dim = np.asarray(model.partition["number_per_dim"], dtype=np.int64)
    logger.info("Growing the tube around the RL rollouts (method: %s)...", cfg.tube_method)
    if cfg.tube_method == "inflation":
        if cfg.inflation_rate is None:
            raise ValueError(
                f"tube_method='inflation' needs {type(model).__name__}.rl_config.inflation_rate "
                f"to be set (one (lower, upper) cell offset per state dimension)."
            )
        active_states = _inflate_cells(newly_visited, cfg.inflation_rate, number_per_dim, model.wrap)
    elif cfg.tube_method == "smart":
        active_states = _smart_inflate_cells(
            visited=newly_visited, model=model, val_env=env,
            actor_critic=actor_critic, params=params,
            discrete_actions=discrete_actions, cfg=cfg, number_per_dim=number_per_dim,
        )
    else:
        raise ValueError(f"Unknown tube_method: {cfg.tube_method}")

    # Obtain the policy (active actions)
    obs_batch = np.asarray(env.obs_low + (active_states.astype(np.float32) + 0.5) * env.bin_widths, dtype=np.float32)
    top_k, rl_policy = find_policy_actions_batch(
        obs_batch, actor_critic, params, discrete_actions, num=cfg.RL_actions_per_state
    )
    active_actions = {tuple(cell): top_k[i] for i, cell in enumerate(active_states.tolist())}
    return active_states, active_actions, rl_policy

__all__ = ["find_active"]
