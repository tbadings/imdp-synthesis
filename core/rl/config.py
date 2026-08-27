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
    distance_cost: float | Sequence[float]
    per_step_cost: float
    eval_steps: int | None = None

    @property
    def rollout_steps(self) -> int:
        return self.eval_steps if self.eval_steps is not None else self.max_steps
