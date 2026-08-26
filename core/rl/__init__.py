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

def _check_reward_balance(cfg: RLConfig, gamma: float):
    """Warn when giving up beats surviving.

    Surviving without reaching the goal costs at most
    (|per_step_reward| + |distance_reward|) / (1 - gamma). If a terminal penalty is
    smaller than that, the best available action from a hard region is to end the
    episode on purpose -- fly into an obstacle or straight out of the state space --
    and the policy converges to doing exactly that.
    """
    living_cost = (abs(cfg.per_step_reward) + abs(cfg.distance_reward)) / max(1.0 - gamma, 1e-9)
    for name, penalty in (("unsafe_penalty", cfg.unsafe_penalty),
                          ("out_of_bounds_penalty", cfg.out_of_bounds_penalty)):
        if abs(penalty) <= living_cost:
            logger.warning(
                "%s=%.3g is not worse than the cost of surviving without reaching the goal "
                "(%.3g). Terminating on purpose is then at least as good as continuing, and the "
                "policy will learn to do so. Use a penalty of magnitude > %.3g.",
                name, penalty, living_cost, living_cost,
            )


def find_active(model, args):
    cfg = RLConfig(
        max_steps=args.max_steps,
        goal_reward=args.goal_reward,
        unsafe_penalty=args.unsafe_penalty,
        out_of_bounds_penalty=args.out_of_bounds_penalty,
        distance_reward=args.distance_reward,
        per_step_reward=args.per_step_reward,
        eval_steps=getattr(args, "eval_steps", None),
        train_noise_factor=args.train_noise_factor,
        critical_margin=args.critical_margin,
    )
    _check_reward_balance(cfg, args.gamma)
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
