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
    # Horizon for evaluation / tube rollouts. Kept separate from `max_steps` because the
    # training truncation is a credit-assignment knob, while this one has to be long
    # enough for the policy to actually reach the goal from x0.
    eval_steps: int | None = None
    # Training-time conservatism. Both deliberately differ from evaluation: the policy is
    # trained against inflated noise and a slightly enlarged obstacle set so that it keeps
    # a margin, while evaluation uses the true noise and the true sets.
    train_noise_factor: float = 2.0
    critical_margin: float = 0.5

    @property
    def rollout_steps(self) -> int:
        return self.eval_steps if self.eval_steps is not None else self.max_steps
