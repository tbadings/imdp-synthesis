import itertools
import logging
from time import time
import numpy as np

from .config import RLConfig
from .env import BenchmarkEnv
from .evaluation import evaluate_policy
from .policy import find_policy_actions_batch
from .ppo import train_ppo
from .tube import _inflate_cells, _smart_inflate_cells

logger = logging.getLogger(__name__)

# Reward terms a benchmark may pin via its `rl_reward` attribute.
REWARD_FIELDS = (
    "goal_reward",
    "unsafe_penalty",
    "out_of_bounds_penalty",
    "distance_reward",
    "per_step_reward",
)


def _resolve_reward(model, args):
    """Merge the reward terms, lowest precedence first.

    1. the ``--*_reward`` / ``--*_penalty`` defaults from `core.options`,
    2. whatever the benchmark pins in its `rl_reward` attribute,
    3. options actually given on the command line, which always win.

    A benchmark that leaves a term out simply keeps the default for it.
    """
    resolved = {field: getattr(args, field) for field in REWARD_FIELDS}

    overrides = getattr(model, "rl_reward", None) or {}
    unknown = set(overrides) - set(REWARD_FIELDS)
    if unknown:
        raise ValueError(
            f"{type(model).__name__}.rl_reward has unknown entries {sorted(unknown)}; "
            f"expected any of {list(REWARD_FIELDS)}"
        )

    # `cli_provided` is absent when args is built by hand rather than parsed; then nothing is
    # treated as explicitly given and the benchmark's values apply.
    provided = getattr(args, "cli_provided", frozenset())
    for field, value in overrides.items():
        if field in provided:
            logger.info("- Reward: --%s given on the command line, overriding %s.rl_reward",
                        field, type(model).__name__)
            continue
        resolved[field] = value

    return resolved


def find_active(model, args):
    cfg = RLConfig(
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        **_resolve_reward(model, args),
    )
    env = BenchmarkEnv(model, cfg)

    actor_critic, params = train_ppo(
        env=env,
        args=args,
        pi_arch=tuple(args.pi_arch if args.pi_arch is not None else model.pi_arch),
        vf_arch=tuple(args.vf_arch if args.vf_arch is not None else model.vf_arch),
        seed=args.seed,
    )

    # Discretize continuous control space
    discrete_actions_per_dim = [
        np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i])
        for i in range(len(model.num_actions))
    ]
    discrete_actions = np.array(list(itertools.product(*discrete_actions_per_dim)), dtype=np.float32)

    # Policy evaluation rollouts
    goal_reached, newly_visited, _ = evaluate_policy(
        actor_critic=actor_critic, params=params,
        base_model=model, env=env, cfg=cfg, episodes=args.eval_episodes,
        dims=list(model.plot_dimensions), args=args,
        discrete_actions=discrete_actions, seed=args.seed,
    )
    logger.info(f"Goal reached in {goal_reached}/{args.eval_episodes} episodes.")

    # Compute the tube (active states)
    number_per_dim = np.asarray(model.partition["number_per_dim"], dtype=np.int64)
    if args.tube_method == "inflation":
        active_states = _inflate_cells(newly_visited, model.inflation_rate, number_per_dim)
    elif args.tube_method == "smart":
        active_states = _smart_inflate_cells(
            visited=newly_visited, model=model, val_env=env,
            actor_critic=actor_critic, params=params,
            discrete_actions=discrete_actions, args=args, number_per_dim=number_per_dim,
            noise_support_ratio=args.smart_tube_rate,
        )
    else:
        raise ValueError(f"Unknown tube_method: {args.tube_method}")

    # Obtain the policy (active actions)
    obs_batch = np.asarray(env.obs_low + (active_states.astype(np.float32) + 0.5) * env.bin_widths, dtype=np.float32)
    top_k, rl_policy = find_policy_actions_batch(
        obs_batch, actor_critic, params, discrete_actions, num=args.RL_actions_per_state
    )
    active_actions = {tuple(cell): top_k[i] for i, cell in enumerate(active_states.tolist())}
    return active_states, active_actions, rl_policy

__all__ = ["find_active"]
