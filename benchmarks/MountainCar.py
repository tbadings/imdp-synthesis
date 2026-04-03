from functools import partial
from benchmarks.models import MountainCarDynamics
import jax
import jax.numpy as jnp
import numpy as np
from benchmarks.dynamics import setmath
from matplotlib import animation
import numpy as onp
from matplotlib.patches import Polygon, Circle
import math
from matplotlib import pyplot as plt


class MountainCar(MountainCarDynamics):
    '''
    Pendulum benchmark.
    '''

    def __init__(self, args):
        MountainCarDynamics.__init__(self, args)

        self.plot_dimensions = [0, 1]

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 1

        self.set_spec()
        print('')

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}
        
        # Authority limit for the control u, both positive and negative
        self.uMin = [-1]
        self.uMax = [1]
        self.num_actions = [5]

        self.partition['boundary'] = np.array([[-1.2, -0.07], [0.6, 0.07]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = 0.5 * np.array([360, 140])

        self.goal = np.array([
            [[0.45, -0.07], [0.6, 0.07]]
        ], dtype=float)

        self.critical = np.array([
        ], dtype=float)

        self.x0 = np.array([-0.6, 0])

        return
    
    def plot_trajectory_gif(self, positions, filename="mountaincar.gif", fps=30, dpi=300):
        """
        Create and export a MountainCar-like GIF given a 1D array of positions.
        """

        xs = onp.asarray(positions, dtype=float).ravel()
        if xs.size == 0:
            raise ValueError("positions must be a non-empty 1D array.")

        xmin, xmax = -1.2, 0.6
        goal_x = 0.5
        xs = onp.clip(xs, xmin, xmax)

        def hill(x):
            return onp.sin(3.0 * x) * 0.45 + 0.55

        def slope(x):
            return 3.0 * onp.cos(3.0 * x) * 0.45

        # Track
        track_x = onp.linspace(xmin, xmax, 1000)
        track_y = hill(track_x)
        y_min = track_y.min() - 0.25
        y_max = track_y.max() + 0.25

        # Figure
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("MountainCar")

        # Ground and track
        ax.fill_between(track_x, y_min, track_y, color=(0.62, 0.51, 0.40), zorder=0)
        ax.plot(track_x, track_y, color="black", linewidth=1.5, zorder=1)

        # Goal flag
        fx = float(onp.clip(goal_x, xmin, xmax))
        fy = float(hill(fx))
        flagpole_h = 0.12
        ax.plot([fx, fx], [fy, fy + flagpole_h], color="black", linewidth=2, zorder=2)
        flag_tri = onp.array([
            [fx, fy + flagpole_h],
            [fx, fy + flagpole_h - 0.03],
            [fx + 0.05, fy + flagpole_h - 0.015],
        ])
        ax.add_patch(Polygon(flag_tri, closed=True, facecolor=(0.85, 0.25, 0.25), edgecolor="black", zorder=3))

        # Car geometry and patches
        car_w, car_h = 0.1, 0.06
        wheel_r = 0.02
        car_poly = Polygon(onp.zeros((4, 2)), closed=True, facecolor="black", edgecolor="black", linewidth=1.0, zorder=4)
        wheel_l = Circle((0, 0), wheel_r, facecolor="gray", edgecolor="gray", zorder=5)
        wheel_rp = Circle((0, 0), wheel_r, facecolor="gray", edgecolor="gray", zorder=5)
        ax.add_patch(car_poly)
        ax.add_patch(wheel_l)
        ax.add_patch(wheel_rp)

        fig.tight_layout()

        def car_pose(x):
            th = math.atan(slope(x))
            ct, st = math.cos(th), math.sin(th)
            # Normal offset to keep the car resting on the surface
            n_hat = onp.array([-st, ct])
            center = onp.array([x, hill(x)]) + n_hat * (wheel_r + car_h / 2.0)
            R = onp.array([[ct, -st], [st, ct]])
            # Car body rectangle (centered)
            rect_local = onp.array([
                [-car_w / 2, -car_h / 2],
                [ car_w / 2, -car_h / 2],
                [ car_w / 2,  car_h / 2],
                [-car_w / 2,  car_h / 2],
            ])
            rect_world = rect_local @ R.T + center
            # Wheels at the bottom corners in local frame
            wl_local = onp.array([-car_w * 0.25, -car_h / 2.0])
            wr_local = onp.array([ car_w * 0.25, -car_h / 2.0])
            wl = wl_local @ R.T + center
            wr = wr_local @ R.T + center
            return rect_world, wl, wr

        def init():
            v, wl, wr = car_pose(xs[0])
            car_poly.set_xy(v)
            wheel_l.center = tuple(wl)
            wheel_rp.center = tuple(wr)
            return [car_poly, wheel_l, wheel_rp]

        def update(i):
            v, wl, wr = car_pose(xs[i])
            car_poly.set_xy(v)
            wheel_l.center = tuple(wl)
            wheel_rp.center = tuple(wr)
            return [car_poly, wheel_l, wheel_rp]

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(xs),
            init_func=init,
            blit=True,
            interval=1000.0 / max(1, fps),
        )
        writer = animation.PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        plt.close(fig)