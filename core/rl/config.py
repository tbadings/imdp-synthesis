from dataclasses import dataclass

# Batch chunk size for forward reachable set expansions
CHUNK_SIZE = 16384

# Configuration for RL training
@dataclass
class RLConfig:
    max_steps: int
    goal_reward: float
    unsafe_penalty: float
    out_of_bounds_penalty: float
    distance_reward: float
    per_step_reward: float
