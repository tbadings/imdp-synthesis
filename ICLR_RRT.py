"""Run RRT-guided SVMDP experiments."""

from ICLR_Q1 import run_fixed_SVMDP

if __name__ == "__main__":
    for tube_radius in ((5, 5, 5, 5), (6, 6, 6, 6), (7, 7, 7, 7), (8, 8, 8, 8), (9, 9, 9, 9), (10, 10, 10, 10)):
        for seed in range(5):
            run_fixed_SVMDP(
                args=["--model", "Dubins4D", "--rrt", "--solver", "jax", "--satprob", "0.99", "--seed", str(seed)],
                rrt=dict(
                    iterations=15000,
                    step_horizon=1,
                    control_samples=16,
                    parent_candidates=16,
                    goal_bias=0.1,
                    time_weight=0.01,
                    tube_radius=tube_radius,
                    clearance=(2,),
                ),
            )

    for tube_radius in ((4, 2, 4, 2), (5, 3, 5, 3), (6, 4, 6, 4), (7, 5, 7, 5), (8, 6, 8, 6)):
        for seed in range(5):
            run_fixed_SVMDP(
                args=["--model", "Drone4D", "--rrt", "--solver", "jax", "--satprob", "0.99", "--seed", str(seed)],
                rrt=dict(
                    iterations=15000,
                    step_horizon=1,
                    control_samples=16,
                    parent_candidates=16,
                    goal_bias=0.1,
                    time_weight=0.01,
                    tube_radius=tube_radius,
                    clearance=(2,),
                ),
            )
