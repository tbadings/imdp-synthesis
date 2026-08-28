import logging
from dataclasses import dataclass, fields, replace
from typing import Sequence

logger = logging.getLogger(__name__)

# Batch chunk size for forward reachable set expansions
CHUNK_SIZE = 16384


# Configuration for RL training
@dataclass
class RLConfig:
    """Every RL-related setting for one benchmark, in one place.

    A benchmark pins its own values with ``self.rl_config = RLConfig(...)`` in ``set_spec()``
    and inherits the defaults below for whatever it leaves out. The matching command-line
    options in `core.options` all default to ``None`` (= "not given"), so the precedence is

        RLConfig defaults  <  the benchmark's rl_config  <  the command line.

    See `resolve_rl_config`, which performs that merge.
    """

    # --- Reward function -------------------------------------------------------------------
    goal_reward: float = 5.0
    unsafe_penalty: float = -5.0
    out_of_bounds_penalty: float = -5.0
    # Either a single value scaling every state dimension, or one value per state dimension.
    distance_cost: float | Sequence[float] = 0.0
    per_step_cost: float = 0.0
    proximity_penalty: float = 0.0
    proximity_dims: Sequence[int] = ()
    # --- Rollouts --------------------------------------------------------------------------
    # Truncation for *training* episodes; eval_steps defaults to it when left at None.
    max_steps: int = 128
    eval_steps: int | None = None
    eval_episodes: int = 2500

    # --- PPO -------------------------------------------------------------------------------
    total_timesteps: int = 200000
    learning_rate: float = 3e-4
    ent_coef: float = 0.005
    rl_batch_size: int = 1024
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
    finetune_steps: int = 0
    subproc: bool = False

    # --- Tube around the RL rollouts -------------------------------------------------------
    RL_actions_per_state: int = 4
    tube_method: str = "inflation"
    # Cells to grow the tube by per state dimension, as (lower, upper) offsets. Only used by
    # tube_method='inflation', which is why it has no all-benchmarks default.
    inflation_rate: Sequence[tuple[int, int]] | None = None
    smart_tube_rate: float = 0.5

    @property
    def rollout_steps(self) -> int:
        return self.eval_steps if self.eval_steps is not None else self.max_steps


# The command-line options in `core.options` that map onto an RLConfig field, by field name.
RL_FIELDS = tuple(f.name for f in fields(RLConfig))


def resolve_rl_config(model, args) -> RLConfig:
    """Merge the benchmark's `rl_config` with the RL options given on the command line.

    Every RL option defaults to None in `core.options`, so a non-None value is one the caller
    actually asked for, and those always win over what the benchmark pins.
    """

    cfg = getattr(model, "rl_config", None)
    if cfg is None:
        logger.warning("%s has no rl_config; falling back to the RLConfig defaults.",
                       type(model).__name__)
        cfg = RLConfig()
    elif not isinstance(cfg, RLConfig):
        raise TypeError(
            f"{type(model).__name__}.rl_config must be an RLConfig, got {type(cfg).__name__}"
        )

    overrides = {
        name: getattr(args, name) for name in RL_FIELDS if getattr(args, name, None) is not None
    }
    for name, value in overrides.items():
        logger.info("- RL: --%s=%s given on the command line, overriding %s.rl_config",
                    name, value, type(model).__name__)

    return replace(cfg, **overrides)
