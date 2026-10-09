from functools import partial
from benchmarks.models import DroneDynamics_2agent
import jax
import jax.numpy as jnp
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from benchmarks.dynamics import setmath
from core.plotting.utils import save_fig
from core.rl.config import RLConfig



class Drone4D_2agent(DroneDynamics_2agent):
    '''
    Two independent Drone4D agents, with an 8D state space and a 4D control input space.
    '''

    # Position dimensions (x, y) of each drone, and the colors to plot the drones in
    POSITION_DIMS = [(0, 2), (4, 6)]
    DRONE_COLORS = ['tab:blue', 'tab:orange']

    def __init__(self, args):
        DroneDynamics_2agent.__init__(self, args)

        self.plot_dimensions = [0, 2]

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 1

        self.set_spec()

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-1, -1, -1, -1]
        self.uMax = [1 ,1, 1, 1]
        self.num_actions = [5, 5, 5, 5]

        v_min = self.v_min
        v_max = self.v_max

        self.partition['boundary'] = np.array([[-2.5, v_min, -2.5, v_min, -2.5, v_min, -2.5, v_min],
                                               [2.5, v_max, 2.5, v_max, 2.5, v_max, 2.5, v_max]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([20, 8, 20, 8, 20, 8, 20, 8])

        self.goal = np.array([
            [[-2.5, v_min, -1, v_min, 1.25, v_min, -1, v_min],
             [-1.25, v_max, 1, v_max, 2.5, v_max, 1, v_max]],
            [[1.25, v_min, -1, v_min, -2.5, v_min, -1, v_min],
             [2.5, v_max, 1, v_max, -1.25, v_max, 1, v_max]]
        ], dtype=float)

        # Obstacles in the position plane, as [[x_lo, y_lo], [x_hi, y_hi]]; both drones must avoid them
        self.obstacles = np.array([
            [[-0.5, -0.5], [0.5, 0.5]],
            [[-2.5, 1], [-0.5, 1.5]]
        ], dtype=float)

        # Critical: either drone in an obstacle, or both drones in the same position cell (a collision)
        self.critical = np.concatenate([self.obstacle_boxes(self.obstacles), self.collision_boxes()])

        self.x0 = np.array([-1.5, 0, 2.25, -0.01, 1.5, 0, -2.25, 0.01])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="ppo",
            total_timesteps=5000000,
            RL_actions_per_state=3**4,
            inflation_rate=[(-3, 3), (-1, 1), (-3, 3), (-1, 1), (-3, 3), (-1, 1), (-3, 3), (-1, 1)],
                        # [(-4, 4), (-2, 2), (-4, 4), (-2, 2), (-4, 4), (-2, 2), (-4, 4), (-2, 2)],
            proximity_dims = [0, 2, 4, 6],
            goal_reward=50.0,
            unsafe_penalty=-50.0,
            out_of_bounds_penalty=-50.0,
            distance_cost=[0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0],
            per_step_cost=0.01,
            proximity_penalty=0.1,
            eval_episodes=100,
            pi_arch=[256, 256],
            vf_arch=[256, 256],
        )

        return

    def obstacle_boxes(self, obstacles):
        '''
        Lift obstacles in the position plane to boxes in the state space, one per obstacle and drone.

        The box of a drone restricts that drone's position (x, y) to the obstacle and leaves all other dimensions
        free, so it covers the states where that drone is in the obstacle, wherever the other drone is.

        Args:
            obstacles: np.ndarray of shape (N, 2, 2), the lower and upper (x, y) corners of every obstacle
        '''

        lower, upper = self.partition['boundary']

        boxes = []
        for obstacle in obstacles:
            for dims in self.POSITION_DIMS:
                box = np.array([lower, upper], dtype=float)
                box[:, list(dims)] = obstacle
                boxes.append(box)

        return np.array(boxes).reshape(-1, 2, len(lower))

    def collision_boxes(self):
        '''
        Cover the states where both drones are in the same position cell of the partition grid with boxes.

        There is one box per position cell (an x cell and a y cell): it restricts the x positions of both
        drones (dimensions 0 and 4) to the x cell and their y positions (dimensions 2 and 6) to the y cell,
        and leaves the velocities free. The partition marks a cell as critical iff both drones are in the
        same position cell, as boxes that only touch a cell do not count as overlapping it.
        '''

        lower, upper = self.partition['boundary']
        number = self.partition['number_per_dim']
        for d1, d2 in [(0, 4), (2, 6)]:
            assert lower[d1] == lower[d2] and upper[d1] == upper[d2] and number[d1] == number[d2], \
                f"Both drones need the same position grid in dimensions {d1} and {d2}"

        x_edges = np.linspace(lower[0], upper[0], number[0] + 1)
        y_edges = np.linspace(lower[2], upper[2], number[2] + 1)

        boxes = []
        for x_lo, x_hi in zip(x_edges[:-1], x_edges[1:]):
            for y_lo, y_hi in zip(y_edges[:-1], y_edges[1:]):
                box = np.array([lower, upper], dtype=float)
                box[:, [0, 4]] = [[x_lo], [x_hi]]
                box[:, [2, 6]] = [[y_lo], [y_hi]]
                boxes.append(box)

        return np.array(boxes)

    def _plot_position_plane(self, ax):
        '''
        Draw the position plane shared by both drones: the position cells of the partition, each drone's goal, and
        the obstacles.
        '''

        # The model parser turns the boundary into a JAX array
        lower, upper = np.asarray(self.partition['boundary'])
        number = np.asarray(self.partition['number_per_dim'])

        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[2], upper[2])
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xticks(np.linspace(lower[0], upper[0], number[0] + 1), minor=True)
        ax.set_yticks(np.linspace(lower[2], upper[2], number[2] + 1), minor=True)
        ax.grid(which='minor', color='lightgray', lw=0.5)

        for goal in self.goal:
            for agent, ((ix, iy), color) in enumerate(zip(self.POSITION_DIMS, self.DRONE_COLORS)):
                ax.add_patch(Rectangle((goal[0, ix], goal[0, iy]), goal[1, ix] - goal[0, ix], goal[1, iy] - goal[0, iy],
                                       facecolor=color, edgecolor=color, alpha=0.2, hatch='//', lw=0,
                                       label=f'Goal drone {agent + 1}'))

        for (x_lo, y_lo), (x_hi, y_hi) in self.obstacles:
            ax.add_patch(Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo, facecolor='dimgray', alpha=0.6, lw=0,
                                   label='Obstacle'))

    def plot_trace(self, trajectory, filename):
        '''
        Plots a trajectory of both drones in the position plane, with the time steps of the states.

        Args:
            trajectory: np.ndarray of shape (T, 8), the states of the trajectory
            filename: Output path without extension (stored as pdf and png)
        '''

        trajectory = np.asarray(trajectory, dtype=float)

        fig, ax = plt.subplots(figsize=(6, 6))
        self._plot_position_plane(ax)

        for agent, ((ix, iy), color) in enumerate(zip(self.POSITION_DIMS, self.DRONE_COLORS)):
            ax.plot(trajectory[:, ix], trajectory[:, iy], '-o', color=color, lw=1.5, ms=4, label=f'Drone {agent + 1}')
            ax.plot(trajectory[0, ix], trajectory[0, iy], 's', color=color, ms=8)

            # Consecutive states that barely move (e.g., hovering in the goal) share one label 'first-last'
            positions = trajectory[:, [ix, iy]]
            first = 0
            for k in range(1, len(positions) + 1):
                if k == len(positions) or np.linalg.norm(positions[k] - positions[first]) > 0.15:
                    label = str(first) if k - 1 == first else f'{first}–{k - 1}'
                    ax.annotate(label, positions[first], xytext=(4, 4), textcoords='offset points', fontsize=7,
                                color=color)
                    first = k

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        save_fig(fig, filename)

    def plot_trajectory_gif(self, trajectory, filename="drone4d_2agent_trajectory.gif", substeps=10):
        '''
        Plots a trajectory of both drones as an animation and stores it as a gif.

        Positions are interpolated linearly between time steps, and the trails mark the states at the time steps.
        The position cell of each drone at the last time step is shaded, in red if both drones are in the same cell
        (a collision).

        Args:
            trajectory: np.ndarray of shape (T, 8), the states of the trajectory
            filename: Output filename for the gif
            substeps: Number of frames per time step
        '''

        trajectory = np.asarray(trajectory, dtype=float)
        lower, upper = np.asarray(self.partition['boundary'])
        width = (upper - lower) / np.asarray(self.partition['number_per_dim'])

        fig, ax = plt.subplots(figsize=(6, 6))
        self._plot_position_plane(ax)

        trails, markers, cells = [], [], []
        for agent, ((ix, iy), color) in enumerate(zip(self.POSITION_DIMS, self.DRONE_COLORS)):
            trails.append(ax.plot([], [], '-o', color=color, lw=1.5, ms=4, label=f'Drone {agent + 1}')[0])
            markers.append(ax.plot([], [], 'o', color=color, ms=10)[0])
            cells.append(ax.add_patch(Rectangle((0, 0), width[ix], width[iy], facecolor=color, alpha=0.4)))
        time_text = ax.set_title('step 0')  # Not empty, so that tight_layout leaves room for it
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        fig.tight_layout()

        # Frames between the time steps, plus frames that hold the final state
        num_frames = (len(trajectory) - 1) * substeps + 1
        hold_frames = 2 * substeps

        def animate(i):
            k, sub = divmod(min(i, num_frames - 1), substeps)
            frac = sub / substeps
            state = (1 - frac) * trajectory[k] + frac * trajectory[min(k + 1, len(trajectory) - 1)]

            # Grid cell of the state at the last time step; a collision if both drones share a position cell
            cell_idx = np.floor((trajectory[k] - lower) / width)
            collision = np.array_equal(*(cell_idx[list(dims)] for dims in self.POSITION_DIMS))

            for (ix, iy), color, trail, marker, cell in zip(self.POSITION_DIMS, self.DRONE_COLORS, trails, markers, cells):
                path = np.vstack([trajectory[:k + 1, [ix, iy]], state[[ix, iy]]])
                trail.set_data(path[:, 0], path[:, 1])
                trail.set_markevery(list(range(k + 1)))
                marker.set_data([state[ix]], [state[iy]])
                cell.set_xy(lower[[ix, iy]] + cell_idx[[ix, iy]] * width[[ix, iy]])
                cell.set_facecolor('crimson' if collision else color)
            time_text.set_text(f'step {k}' + (' (collision)' if collision else ''))
            return *trails, *markers, *cells, time_text

        ani = animation.FuncAnimation(
            fig, animate, frames=num_frames + hold_frames, interval=50, blit=True
        )

        ani.save(filename, writer='pillow', dpi=100)
        plt.close(fig)
