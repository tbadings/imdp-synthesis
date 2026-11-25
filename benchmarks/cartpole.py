'''
CartPole benchmarks (as in Gymnasium, with a continuous force): balance the pole upright on the right side of the
track (CartPole), also passing under an obstacle that hangs above the track (CartPole_hard).
'''

import jax.numpy as jnp
from matplotlib import animation
import numpy as np

from benchmarks.base import Benchmark
from benchmarks.dynamics import setmath
from core.box import Box
from core.rl.config import RLConfig


class CartPoleDynamics(Benchmark):
    '''
    CartPole dynamics model, based on the Gymnasium CartPole environment.
    The state consists of the cart position, cart velocity, pole angle, and pole angular velocity.
    The action is a force applied to the cart. The dynamics are nonlinear and include process noise.

    Differences with the original CartPole environment:
    - The time step is 0.02 * 5 seconds instead of 0.02 seconds.
    - Original environment has no process noise.
    - Original environment has a discrete action space (left/right), while this model uses a continuous action space (force).
    '''

    def __init__(self, args):
        self.tau = 0.02 * 5
        self.state_variables = ['position', 'velocity', 'angle', 'angular_velocity']

        # CartPole parameters (Gymnasium CartPole convention)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # Actually half the pole's length
        self.polemass_length = self.masspole * self.length

        # Gaussian noise acts on the cart velocity, triangular noise on the cart position and the pole angle
        self.set_noise(args,
                       gaussian=dict(variance=np.array([0, 0.001, 0, 0])**2, cells=[1, 25, 1, 1]),
                       triangular=dict(halfwidth=[0.005, 0, 0.005, 0], cells=[10, 1, 10, 1]))

    def step(self, state, action, noise):
        x, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]
        force = action[0]

        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        return jnp.array([x, x_dot, theta, theta_dot]) + noise

    def successor_bounds(self, state_min, state_max, action_min, action_max):
        state_min, state_max = setmath.box(jnp.array(state_min), jnp.array(state_max))
        [x_min, xd_min, th_min, thd_min] = state_min
        [x_max, xd_max, th_max, thd_max] = state_max

        action_min, action_max = setmath.box(jnp.array(action_min), jnp.array(action_max))
        F_min = action_min[0]
        F_max = action_max[0]

        # Interval bounds on the trigonometric terms of the pole angle
        s_min, s_max = setmath.sin(th_min, th_max)
        c_min, c_max = setmath.cos(th_min, th_max)

        # theta_dot^2
        thd2_min, thd2_max = setmath.square(thd_min, thd_max)

        # temp = (F + polemass_length * theta_dot^2 * sin(theta)) / total_mass
        # (setmath.mult/div return shape-(1,) arrays, so we squeeze back to scalars)
        q_min, q_max = setmath.mult((thd2_min, thd2_max), (s_min, s_max))
        q_min, q_max = q_min[0], q_max[0]
        temp_min = (F_min + self.polemass_length * q_min) / self.total_mass
        temp_max = (F_max + self.polemass_length * q_max) / self.total_mass

        # thetaacc = (g*sin - cos*temp) / (length * (4/3 - masspole*cos^2/total_mass))
        b_min, b_max = setmath.mult((c_min, c_max), (temp_min, temp_max))
        b_min, b_max = b_min[0], b_max[0]
        num_min = self.gravity * s_min - b_max
        num_max = self.gravity * s_max - b_min

        c2_min, c2_max = setmath.square(c_min, c_max)
        k = self.masspole / self.total_mass
        # Denominator is guaranteed positive (4/3 - k*cos^2 >= 4/3 - k > 0)
        denom_min = self.length * (4.0 / 3.0 - k * c2_max)
        denom_max = self.length * (4.0 / 3.0 - k * c2_min)

        thacc_min, thacc_max = setmath.div((num_min, num_max), (denom_min, denom_max))
        thacc_min, thacc_max = thacc_min[0], thacc_max[0]

        # xacc = temp - polemass_length * thetaacc * cos / total_mass
        e_min, e_max = setmath.mult((thacc_min, thacc_max), (c_min, c_max))
        e_min, e_max = e_min[0], e_max[0]
        coef = self.polemass_length / self.total_mass
        xacc_min = temp_min - coef * e_max
        xacc_max = temp_max - coef * e_min

        # Euler integration
        x_next = jnp.array([x_min + self.tau * xd_min, x_max + self.tau * xd_max])
        xd_next = jnp.array([xd_min + self.tau * xacc_min, xd_max + self.tau * xacc_max])
        th_next = jnp.array([th_min + self.tau * thd_min, th_max + self.tau * thd_max])
        thd_next = jnp.array([thd_min + self.tau * thacc_min, thd_max + self.tau * thacc_max])

        state_next = jnp.vstack((x_next, xd_next, th_next, thd_next))

        return jnp.min(state_next, axis=1), jnp.max(state_next, axis=1)


class CartPole(CartPoleDynamics):
    '''
    CartPole benchmark (cart position, cart velocity, pole angle, pole angular velocity).
    '''

    def __init__(self, args):
        CartPoleDynamics.__init__(self, args)

        # Plot the cart position against the pole angle
        self.plot_dimensions = [0, 2]

        self.state_space = Box([-2.4, -3.0, -0.25, -3.0], [2.4, 3.0, 0.25, 3.0], names=self.state_variables)
        self.state_grid = [200, 200, 200, 200]
        # Force on the cart
        self.input_space = Box([-10], [10])
        self.input_grid = [41]

        # Goal: balance the pole upright on the right side of the track
        self.goal = [self.state_space.restrict(position=(1, 2.4), velocity=(-1.5, 1.5), angle=(-0.05, 0.05),
                                               angular_velocity=(-1.0, 1.0))]
        self.critical = []

        # Start upright on the left side of the track
        self.x0 = np.array([-1, 0.0, 0, 0.0])

        # RL configuration: networks, PPO training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        self.rl_config = RLConfig(
            rl_algo="sac",
            total_timesteps= 5000000,
            goal_reward=20.0,
            unsafe_penalty=-20.0,
            out_of_bounds_penalty=-20.0,
            distance_cost=[0.1, 0.0, 0.0, 0.0],
            per_step_cost=0.05,
            inflation_rate=[(-7, 7), (-7, 7), (-7, 7), (-7, 7)],
            RL_actions_per_state=5,
        )

    def plot_simulation(self, args, stamp, partition, traces):
        print('Plot CartPole gif...')
        self.plot_trajectory_gif(np.array(traces[0]['x'])[:, [0, 2]],
                                 filename=str(args.output_dir / f'cartpole_{stamp}.gif'))

    def plot_trajectory_gif(self, trajectory, filename="cartpole_trajectory.gif"):
        """
        Plots a trajectory of the cartpole as an animation and stores it as a gif.

        Args:
            trajectory: np.ndarray of shape (T, 2), where each row is [cart_position, pole_angle]
            filename: Output filename for the gif
        """
        import matplotlib.pyplot as plt

        trajectory = np.asarray(trajectory, dtype=float)
        x_positions = trajectory[:, 0]
        thetas = trajectory[:, 1]

        x_lim = 2.4
        pole_len = 2 * self.length  # Physical length of the pole (self.length is half of it)
        cart_w, cart_h = 0.4, 0.2
        y_top = cart_h + pole_len + 0.3

        # Prepare figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlim(-x_lim - 1, x_lim + 1)
        ax.set_ylim(-0.5, y_top)
        ax.set_aspect('equal')
        ax.grid()

        # Track
        ax.axhline(0, color='black', lw=1)

        from matplotlib.patches import Rectangle

        # Obstacles hanging from the ceiling; their lower edge is measured from the pole's pivot
        for x_min, x_max, bottom in getattr(self, 'obstacles', []):
            ax.add_patch(Rectangle((x_min, cart_h + bottom), x_max - x_min, y_top - cart_h - bottom,
                                   facecolor='tab:red', edgecolor='darkred', hatch='//', alpha=0.6))

        cart = Rectangle((0, 0), cart_w, cart_h, facecolor='black')
        ax.add_patch(cart)
        pole, = ax.plot([], [], '-', lw=4, color='tab:brown')
        time_template = 'step {:d}'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            cart.set_xy((-cart_w / 2, 0))
            pole.set_data([], [])
            time_text.set_text('')
            return cart, pole, time_text

        def animate(i):
            x = x_positions[i]
            theta = thetas[i]
            cart.set_xy((x - cart_w / 2, 0))
            # Pole pivots at the top-center of the cart; theta=0 is upright
            px = x + pole_len * np.sin(theta)
            py = cart_h + pole_len * np.cos(theta)
            pole.set_data([x, px], [cart_h, py])
            time_text.set_text(time_template.format(i))
            return cart, pole, time_text

        ani = animation.FuncAnimation(
            fig, animate, frames=len(trajectory), interval=50, blit=True, init_func=init
        )

        ani.save(filename, writer='pillow')
        plt.close(fig)


class CartPole_hard(CartPoleDynamics):
    '''
    CartPole with an obstacle hanging above the middle of the track. The upright pole does not fit
    under the obstacle, so the pole must be tilted far enough while the cart passes underneath.
    '''

    # Same animation as CartPole (it draws the obstacles)
    plot_simulation = CartPole.plot_simulation
    plot_trajectory_gif = CartPole.plot_trajectory_gif

    def __init__(self, args):
        CartPoleDynamics.__init__(self, args)

        # Plot the cart position against the pole angle
        self.plot_dimensions = [0, 2]

        # The pole has to tilt beyond the obstacle's clearance angle (~0.14 rad), so the angle range
        # is wider than for CartPole.
        self.state_space = Box([-2.4, -5.0, -0.5, -3.0], [2.4, 5.0, 0.5, 3.0], names=self.state_variables)
        self.state_grid = [200, 400, 400, 200]
        # Force on the cart
        self.input_space = Box([-10], [10])
        self.input_grid = [41]

        # Obstacles hanging from the ceiling, as (x_min, x_max, bottom), where [x_min, x_max] is the
        # horizontal extent and bottom is the height of the lower edge above the pole's pivot.
        self.pole_length = 2 * self.length  # self.length is half the pole's length
        # The pole must tilt past ~8 deg to pass. Wider or lower obstacles need a longer or larger tilt,
        # which SAC does not learn reliably with this reward (e.g. 0.4 m wide at bottom 0.97 never succeeds).
        self.obstacles = [(-0.05, 0.05, 0.97)]

        # Goal: balance the pole upright on the right side of the track, past the obstacle
        self.goal = [self.state_space.restrict(position=(1, 2.4), velocity=(-1.5, 1.5), angle=(-0.05, 0.05),
                                               angular_velocity=(-1.0, 1.0))]

        # Cart position and pole angle for which the pole hits an obstacle, covered by boxes
        self.critical = self.obstacle_boxes(num_bands=32)

        # Start upright on the left side of the obstacle
        self.x0 = np.array([-1, 0.0, 0, 0.0])

        # RL configuration: networks, SAC training, reward function, and the tube
        # grown around the RL rollouts to form the abstraction.
        # Penalties satisfy |penalty| > (per_step_cost + norm(distance_cost)) / (1 - gamma) = 15,
        # so that ending an episode on purpose never beats surviving.
        self.rl_config = RLConfig(
            rl_algo="sac",
            total_timesteps=10000000,
            goal_reward=20.0,
            unsafe_penalty=-35.0,
            out_of_bounds_penalty=-35.0,
            distance_cost=[0.1, 0.0, 0.0, 0.0],
            per_step_cost=0.2,
            inflation_rate=[(-9, 9), (-9, 9), (-9, 9), (-9, 9)],
            RL_actions_per_state=21,
        )

    def obstacle_boxes(self, num_bands):
        '''
        Cover the states where the pole intersects an obstacle with boxes in (position, angle).

        A point at distance s along the pole is at (x + s*sin(angle), s*cos(angle)) relative to the
        pivot. For an obstacle (x_min, x_max, bottom), the part of the pole above the lower edge
        exists iff cos(angle) >= bottom / pole_length, and then spans the horizontal interval
        x + [min(o1, o2), max(o1, o2)] with o1 = pole_length*sin(angle), o2 = bottom*tan(angle).
        The pole thus hits the obstacle iff
            x_min - max(o1, o2) <= x <= x_max - min(o1, o2).
        Both offsets increase with the angle, so over an angle band [a_lo, a_hi] the union of these
        intervals is [x_min - max(o1, o2)(a_hi), x_max - min(o1, o2)(a_lo)], giving one box per band
        that covers the unsafe set exactly within that band.
        '''

        lower, upper = self.state_space.lower, self.state_space.upper
        boxes = []
        for x_min, x_max, bottom in self.obstacles:
            if bottom >= self.pole_length:
                continue
            clearance = np.arccos(bottom / self.pole_length)
            edges = np.linspace(-clearance, clearance, num_bands + 1)
            for a_lo, a_hi in zip(edges[:-1], edges[1:]):
                offset_hi = max(self.pole_length * np.sin(a_hi), bottom * np.tan(a_hi))
                offset_lo = min(self.pole_length * np.sin(a_lo), bottom * np.tan(a_lo))
                position = (max(x_min - offset_hi, lower[0]), min(x_max - offset_lo, upper[0]))
                boxes.append(self.state_space.restrict(position=position, angle=(a_lo, a_hi)))

        return boxes
