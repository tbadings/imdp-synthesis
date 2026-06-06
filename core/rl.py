import multiprocessing
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import itertools
import benchmarks

@dataclass
class RLConfig:
    max_steps: int
    goal_reward: float
    unsafe_penalty: float
    out_of_bounds_penalty: float
    revisit_penalty: float

class BenchmarkRLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model, cfg: RLConfig, previous_cells=None):
        super().__init__()
        self.model = model
        self.cfg = cfg

        boundary = np.asarray(self.model.partition["boundary"], dtype=np.float32)
        self.obs_low = boundary[0]
        self.obs_high = boundary[1]

        self.observation_space = spaces.Box(self.obs_low, self.obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.asarray(self.model.uMin, dtype=np.float32),
            high=np.asarray(self.model.uMax, dtype=np.float32),
            dtype=np.float32,
        )

        self.bin_widths = (self.obs_high - self.obs_low) / self.model.partition['number_per_dim']
        self.goal = np.asarray(getattr(self.model, "goal", np.empty((0, 2, self.model.n))), dtype=np.float32)
        self.critical = np.asarray(
            getattr(self.model, "critical", np.empty((0, 2, self.model.n))), dtype=np.float32
        )

        self.previous_cells = set() if previous_cells is None else set(previous_cells)
        self.state = None
        self.steps = 0
        self.prev_dist = None

    def set_previous_cells(self, previous_cells):
        self.previous_cells = set(previous_cells)

    def state_to_cell(self, obs):
        indices = np.floor((np.asarray(obs, dtype=np.float64) - self.obs_low) / self.bin_widths).astype(int)
        return tuple(np.clip(indices, 0, self.model.partition['number_per_dim'] - 1).tolist())

    def _in_boxes(self, state, boxes):
        if boxes.size == 0:
            return False
        mins = boxes[:, 0, :]
        maxs = boxes[:, 1, :]
        return bool(np.any(np.all((state >= mins) & (state <= maxs), axis=1)))

    def _goal_center(self):
        if self.goal.size == 0:
            return None
        first_goal = self.goal[0]
        return 0.5 * (first_goal[0] + first_goal[1])

    def _progress_reward(self, state):
        center = self._goal_center()
        if center is None:
            return 0.0
        dist = float(np.linalg.norm(state - center))
        reward = (self.prev_dist - dist) / max(float(np.linalg.norm(self.obs_high - self.obs_low)), 1e-6)
        self.prev_dist = dist
        return reward

    def _wrap_periodic_dims(self, state):
        wrap = np.asarray(getattr(self.model, "wrap", np.zeros(self.model.n, dtype=bool)), dtype=bool)
        if not np.any(wrap):
            return state
        wrapped = state.copy()
        lengths = self.obs_high - self.obs_low
        periodic_idx = np.where(wrap)[0]
        for idx in periodic_idx:
            length = lengths[idx]
            if length <= 0:
                continue
            wrapped[idx] = ((wrapped[idx] - self.obs_low[idx]) % length) + self.obs_low[idx]
        return wrapped

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        testing = bool(options.get("testing", False)) if options else False

        if testing:
            cell = self.state_to_cell(self.model.x0)
            cell_lb = self.obs_low + np.asarray(cell, dtype=np.float32) * self.bin_widths
            cell_ub = cell_lb + self.bin_widths

            # Add a small epsilon, to make sure we appropriately cover the initial state cell
            eps = 0.25 * self.bin_widths
            state = self.np_random.uniform(cell_lb - eps, cell_ub + eps).astype(np.float32)
        else:
            state = self.np_random.uniform(self.obs_low, self.obs_high).astype(np.float32)

        self.state = state
        self.steps = 0

        center = self._goal_center()
        if center is not None:
            self.prev_dist = float(np.linalg.norm(self.state - center))
        else:
            self.prev_dist = 0.0

        return self.state.copy(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)

        noise = np.asarray(self.model.noise.sample(), dtype=np.float32)
        next_state = np.asarray(self.model.step(self.state, action, noise), dtype=np.float32)
        # next_state = self._wrap_periodic_dims(next_state)

        self.state = next_state
        self.steps += 1

        in_goal = self._in_boxes(self.state, self.goal)
        in_critical = self._in_boxes(self.state, self.critical)
        out_of_bounds = bool(np.any(self.state < self.obs_low) or np.any(self.state > self.obs_high))

        if in_goal:
            reward = self.cfg.goal_reward
        elif in_critical:
            reward = self.cfg.unsafe_penalty
        elif out_of_bounds:
            reward = self.cfg.out_of_bounds_penalty
        else:
            reward = self._progress_reward(self.state) - 0.1

        cell = self.state_to_cell(self.state)
        flat_idx = np.ravel_multi_index(cell, self.model.partition['number_per_dim'])
        if flat_idx in self.previous_cells:
            reward -= self.cfg.revisit_penalty

        terminated = in_goal or in_critical or out_of_bounds
        truncated = self.steps >= self.cfg.max_steps

        info = {
            "in_goal": in_goal,
            "in_critical": in_critical,
            "out_of_bounds": out_of_bounds,
            "cell": cell,
        }
        return self.state.copy(), float(reward), terminated, truncated, info

def evaluate_policy(model, norm_env, base_model, cfg, episodes, dims, discrete_actions=None):
    norm_env.training = False
    norm_env.norm_reward = False

    eval_env = BenchmarkRLEnv(base_model, cfg)
    reached_goal = 0
    visited_cells = set()
    trajectories = []

    if discrete_actions is not None:
        discrete_actions = np.asarray(discrete_actions, dtype=np.float32)

    for _ in range(episodes):
        obs, _ = eval_env.reset(options={"testing": True})
        visited_cells.add(eval_env.state_to_cell(obs))

        trace = [obs.copy()]
        for _ in range(cfg.max_steps):
            # Quantize observation to the center of its partition cell.
            if discrete_actions is not None:
                cell = eval_env.state_to_cell(obs)
                obs_q = (eval_env.obs_low + (np.asarray(cell, dtype=np.float32) + 0.5) * eval_env.bin_widths)
                obs_q = np.clip(obs_q, eval_env.obs_low, eval_env.obs_high)

                # print(f"Original obs: {obs}, quantized obs: {obs_q}, cell: {cell}")
            else:
                obs_q = obs

            norm_obs = norm_env.normalize_obs(np.expand_dims(obs_q, axis=0))[0]
            action, _ = model.predict(norm_obs, deterministic=True)

            # Quantize the continuous action to the nearest discrete action.
            if discrete_actions is not None:
                dists = np.linalg.norm(discrete_actions - action, axis=1)
                action = discrete_actions[np.argmin(dists)]

                # print(f"Original action: {action}, quantized action: {action}")

            obs, _, terminated, truncated, info = eval_env.step(action)

            visited_cells.add(info["cell"])
            trace.append(obs.copy())

            if terminated or truncated:
                if terminated and info["in_goal"]:
                    reached_goal += 1
                break

        trajectories.append(np.asarray(trace))

    if len(dims) != 2:
        raise ValueError("This runner currently supports plotting exactly 2 dimensions.")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    if eval_env.critical.size > 0:
        for idx, box in enumerate(eval_env.critical):
            x0, x1 = box[0, dims[0]], box[1, dims[0]]
            y0, y1 = box[0, dims[1]], box[1, dims[1]]
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    color="red",
                    alpha=0.15,
                    label="Critical" if idx == 0 else None,
                )
            )

    if eval_env.goal.size > 0:
        for idx, box in enumerate(eval_env.goal):
            x0, x1 = box[0, dims[0]], box[1, dims[0]]
            y0, y1 = box[0, dims[1]], box[1, dims[1]]
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    color="green",
                    alpha=0.25,
                    label="Goal" if idx == 0 else None,
                )
            )

    for trace in trajectories:
        ax.plot(trace[:, dims[0]], trace[:, dims[1]], linewidth=1.0, alpha=0.9, color="black")

    ax.set_xlim(eval_env.obs_low[dims[0]], eval_env.obs_high[dims[0]])
    ax.set_ylim(eval_env.obs_low[dims[1]], eval_env.obs_high[dims[1]])
    ax.set_xlabel(base_model.state_variables[dims[0]])
    ax.set_ylabel(base_model.state_variables[dims[1]])
    ax.set_title(f"PPO trajectories ({base_model.__class__.__name__})")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    total_cells = int(np.prod(base_model.partition['number_per_dim']))
    return reached_goal, visited_cells, total_cells

def _build_vec_env(base_model, cfg, n_envs, use_subproc, previous_cells):
    env_kwargs = {"model": base_model, "cfg": cfg, "previous_cells": previous_cells}
    vec_env_cls = SubprocVecEnv if use_subproc else DummyVecEnv
    vec_env = make_vec_env(BenchmarkRLEnv, n_envs=n_envs, env_kwargs=env_kwargs, vec_env_cls=vec_env_cls)
    return VecNormalize(vec_env, norm_obs=True, norm_reward=True)

def find_active(model, args, previous_cells):
    cfg = RLConfig(
        max_steps=args.max_steps,
        goal_reward=args.goal_reward,
        unsafe_penalty=args.unsafe_penalty,
        out_of_bounds_penalty=args.out_of_bounds_penalty,
        revisit_penalty=args.revisit_penalty,
    )    

    vec_env = _build_vec_env(
        base_model=model,
        cfg=cfg,
        n_envs=args.n_envs,
        use_subproc=args.subproc,
        previous_cells=previous_cells,
    )
    import torch as th
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))

    ppo = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        n_steps=max(2048 // args.n_envs, 128),
        seed=args.seed,
    )

    ppo.learn(total_timesteps=args.total_timesteps, progress_bar=True)

    discrete_actions_per_dim = [
        np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i])
        for i in range(len(model.num_actions))
    ]
    discrete_actions = np.array(list(itertools.product(*discrete_actions_per_dim)), dtype=np.float32)

    goal_reached, newly_visited, total_cells = evaluate_policy(
        model=ppo,
        norm_env=vec_env,
        base_model=model,
        cfg=cfg,
        episodes=args.eval_episodes,
        dims=list(model.plot_dimensions),
        discrete_actions=discrete_actions,
    )

    print (f"Goal reached in {goal_reached}/{args.eval_episodes} episodes.")

    active_states = set()
    number_per_dim = np.asarray(model.partition['number_per_dim'], dtype=int)

    #inflating visited cells to include neighbors within a certain radius to account for discretization errors and encourage exploration of nearby states
    rate = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    for cell in newly_visited:
        ranges = [range(c + int(lo), c + int(hi) + 1) for c, (lo, hi) in zip(cell, rate)]
        # print(f"Cell {cell} with neighbors {list(itertools.product(*ranges))}")
        for neighbor in itertools.product(*ranges):
            neighbor = tuple(int(v) for v in neighbor)
            valid = True
            for i, val in enumerate(neighbor):
                limit = int(number_per_dim[i])
                if val < 0 or val >= limit:
                    valid = False
                    break
            if valid:
                active_states.add(neighbor)

    active_states = np.array(list(active_states), dtype=int)
    return active_states
