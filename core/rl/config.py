from dataclasses import dataclass
from typing import Sequence

# Batch chunk size for forward reachable set expansions
CHUNK_SIZE = 16384

# Configuration for RL training
@dataclass
class RLConfig:
    max_steps: int
    goal_reward: float
    unsafe_penalty: float
    out_of_bounds_penalty: float
    # Costs, i.e. positive values make surviving without reaching the goal more expensive.
    # `distance_cost` is a scalar, or one weight per state dimension (a benchmark can weight
    # position but not velocity); a scalar is broadcast across all dimensions.
    distance_cost: float | Sequence[float]
    per_step_cost: float
    # Rollout horizon for evaluation / tube construction. Separate from `max_steps`, which
    # truncates *training* episodes: short training episodes reset often and spread the data
    # over the state space, but the rollouts still need room to reach the goal.
    eval_steps: int | None = None

    @property
    def rollout_steps(self) -> int:
        return self.eval_steps if self.eval_steps is not None else self.max_steps
