from functools import partial
from benchmarks.models import PendulumDynamics
import jax
import jax.numpy as jnp
import numpy as np
from benchmarks.dynamics import setmath
from matplotlib import animation


class Pendulum(PendulumDynamics):
    '''
    Pendulum benchmark.
    '''

    def __init__(self, args):
        PendulumDynamics.__init__(self, args)

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
        self.uMin = [-2]
        self.uMax = [2]
        self.num_actions = [7]

        self.partition['boundary'] = np.array([[-np.pi, -8], [np.pi, 8]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = 0.5 * np.array([200, 100])

        self.goal = np.array([
            [[-0.1*np.pi, -1], [0.1*np.pi, 1]]
        ], dtype=float)

        self.critical = np.array([
        ], dtype=float)

        self.x0 = np.array([0.99*np.pi, 0])

        return


    def plot_trajectory_gif(self, trajectory, filename="pendulum_trajectory.gif"):
        """
        Plots a trajectory of the pendulum as an animation and stores it as a gif.

        Args:
            trajectory: np.ndarray of shape (T, 2), where each row is [theta, theta_dot]
            filename: Output filename for the gif
        """
        import matplotlib.pyplot as plt

        # Prepare figure
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=5)
        time_template = 'step {:d}'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        L = 1.0  # Assume pendulum length is 1

        # fig.tight_layout()

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            theta = trajectory[i]
            x = L * np.sin(theta)
            y = L * np.cos(theta)  # Flip the sign to invert the y-axis
            line.set_data([0, x], [0, y])
            time_text.set_text(time_template.format(i))
            return line, time_text

        ani = animation.FuncAnimation(
            fig, animate, frames=len(trajectory), interval=50, blit=True, init_func=init
        )

        ani.save(filename, writer='pillow')
        plt.close(fig)