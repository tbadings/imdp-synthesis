import logging
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration settings for RL training and state space tube construction."""

    # Algorithm selection
    rl_algo: str = "ppo"  # "ppo" or "sac"

    # Reward function
    goal_reward: float = 5.0
    unsafe_penalty: float = -5.0
    out_of_bounds_penalty: float = -5.0
    distance_cost: float | Sequence[float] = 0.0
    per_step_cost: float = 0.0
    proximity_penalty: float = 0.0
    proximity_dims: Sequence[int] = ()

    # Rollouts
    max_steps: int = 128
    eval_steps: int | None = None
    eval_episodes: int = 25000

    # Common & PPO Hyperparameters
    total_timesteps: int = 200000
    learning_rate: float = 3e-4
    ent_coef: float = 0.005
    rl_batch_size: int = 128
    n_steps: int = 128
    n_envs: int = 32
    update_epochs: int = 10
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    adam_eps: float = 1e-5
    pi_arch: Sequence[int] = (64, 64)
    vf_arch: Sequence[int] = (64, 64)

    # SAC Specific Hyperparameters
    buffer_size: int = 65536
    sac_batch_size: int = 128
    warmup_steps: int = 1024
    tau: float = 0.01
    min_alpha: float = 0.05

    # Tube around RL rollouts
    RL_actions_per_state: int = 3
    tube_method: str = "inflation"  # "inflation" or "smart"
    inflation_rate: Sequence[tuple[int, int]] | None = None
    smart_tube_rate: float = 0.75

    # Output and artifact paths
    output_dir: str | Path | None = None

    @property
    def rollout_steps(self) -> int:
        return self.eval_steps if self.eval_steps is not None else self.max_steps


RL_FIELDS = tuple(f.name for f in fields(RLConfig))


def resolve_rl_config(model, args) -> RLConfig:
    """Merge model.rl_config with CLI argument overrides."""
    base_cfg = getattr(model, "rl_config", None) or RLConfig()
    overrides = {name: getattr(args, name) for name in RL_FIELDS if getattr(args, name, None) is not None}
    resolved = replace(base_cfg, **overrides)
    logger.info("Resolved RL config (algorithm: %s): %s", resolved.rl_algo, resolved)
    return resolved
