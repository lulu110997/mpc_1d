from matplotlib import pyplot as plt
"""
Compares CBF, APF and no OA in a situation where the obstacle is static and dynamic
Does not take into account the area of the agents and obstacles (ie only point to point distance)

Author: Louis Fernandez
Contact: louis.f.fernandez@student.uts.edu.au
Date: 2024-03-22
Version: 1.0.0
Credits: 
    https://medium.com/@rymshasiddiqui/path-planning-using-potential-field-algorithm-a30ad12bdb08 for APF explanation
"""
from matplotlib.animation import FuncAnimation
from roboticstoolbox.tools import trajectory as traj
from spatialmath import SE3
from qpsolvers import solve_ls
import math
import numpy as np
import sympy as sym


# Plots
SAVE_ANIMATION = False  # If you want to save an animation
F_NAME = "static"  # Name of the gif animation
SKIP_STEPS = 3  # Useful for skipping steps during simulation to increase speed

STATIC = False; START = 100; END = 500
# STATIC = True; START = 0; END = 500; STATIC_OBS_POS = (0.4, 0.2)

# CBF param
CBF_GAMMA = 1.4  # Represents how close the agent can come to the obstacle (higher = closer)
# CBF_GAMMA = 0.1  # Represents how close the agent can come to the obstacle (higher = closer)

# APF params
APF_Q = 0.1  # Minimum allowable distance before the potential field is activated
NETA = 0.2  # Gain for the repulsive field

# P controller used in cbf and apf
KV = 1  # Velocity gain used when calculating velocity
MAX_VEL = 1.5  # Maximum allowable velocity
MIN_VEL = -1.5  # Minimum allowable velocity

TIME = 1  # Desired time to complete the task

# Radius of ee and obstacle
R_OBS = 0.05
R_EE = 0.05

########################################################################################################################
if not SAVE_ANIMATION:
    plt.ion()

FREQ = 500  # Control frequency of the universal robot E series is 500Hz
DELTA_TIME = 1/FREQ  # Time period
N_STEPS = int(TIME/DELTA_TIME)  # Total number of steps for the simulation


def distance_derivative():
    """
    Symbolically compute distance and its derivative
    Returns: tuple | distance_function and distance_derivative_function
    """
    x_ee = sym.symbols('x_ee:2')
    x_obs = sym.symbols('x_obs:2')
    x_ee = sym.Matrix([[x_ee[0], x_ee[1]]])
    x_obs = sym.Matrix([[x_obs[0], x_obs[1]]])
    R_ee = sym.Symbol('R_ee')
    R_obs = sym.Symbol('R_obs')

    # This provides a much more complicated hx_dot
    alt_hx = sym.sqrt(
        (x_ee[0] - x_obs[0])**2 +
        (x_ee[1] - x_obs[1])**2
    ) - (R_obs + R_ee)

    hx_dot = sym.diff(alt_hx, x_ee)

    hx = sym.lambdify([x_ee, x_obs, R_ee, R_obs], expr=alt_hx)
    hx_dot = sym.lambdify([x_ee, x_obs, R_ee, R_obs], expr=hx_dot)
    return hx, hx_dot

# Obtain the distance function and its derivative
dist_f, dist_deriv = distance_derivative()


def get_vel(x0, x1):
    """
    Obtain the agent's velocity using a discretisation method
    Args:
        x0: array-like | current position
        x1: array-like | desired position

    Returns: tuple | next position and velocity
    """
    assert x0.ndim == 1 and x1.ndim == 1
    vel = KV*((x1 - x0) / DELTA_TIME)  # Need to know time between snapshots. Time it took from pos1 to pos2. This is actual velocity
    # P controller not computing velocity, just a velocity command. The velocity is a positional error (where I want my
    # robot to be based on where I am).
    # Velocity command is proportional to error.
    # PI for first order system (eg velocity)
    # PD for second order system (eg torque commands)

    return vel


def get_force(r, o):
    """
    Obtain the repuilsive force using simple APF
    https://medium.com/@rymshasiddiqui/path-planning-using-potential-field-algorithm-a30ad12bdb08
    Args:
        r: ndarray | current position of robot
        o: ndarray | current position of obstacle

    Returns: tuple | the robot's next position and calculated velocity
    """
    assert r.ndim == 1 and o.ndim == 1
    curr_dist = dist_f(r, o, R_EE, R_OBS)
    if curr_dist < APF_Q:
        nabla_d = dist_deriv(r, o, R_EE, R_OBS).flatten()
        force = -NETA * (1/APF_Q - 1/curr_dist) * (1/curr_dist)**2 * nabla_d
    else:
        force = np.array([0, 0])

    return force


def use_cbf(r, desired_pos, o):
    """
    Use CBF constraint and QP solver to obtain a desired velocity
    Args:
        r: np ndarray | contains the robots current xy position
        desired_pos: np ndarray | contains the desired robots xy position
        o: np ndarray | contains the obstacle's current xy position
    Returns: np ndarray | contains xy velocity

    """
    R = np.identity(2)  # Weights?
    s = KV*((desired_pos - r) / DELTA_TIME)  # What we are trying to optimise
    G = -dist_deriv(r, o, R_EE, R_OBS)  # CBF derivative
    h = np.array([CBF_GAMMA * dist_f(r, o, R_EE, R_OBS)])  # CBF exponential gamma*hx

    vel = solve_ls(R, s, G, h, lb=np.array([MIN_VEL, MIN_VEL]), ub=np.array([MAX_VEL, MAX_VEL]), solver="clarabel")  # osqp or clarabel or cvxopt
    return vel


def cap_vel(vel):
    """
    Caps the xy velocity to max vel
    Args:
        vel: np ndarray | contains xy velocity

    Returns: np ndarray | capped velocity
    """
    assert vel.ndim == 1
    if math.fabs(vel[0]) > MAX_VEL:
        vel[0] = math.copysign(MAX_VEL, vel[0])
    if math.fabs(vel[1]) > MAX_VEL:
        vel[1] = math.copysign(MAX_VEL, vel[1])
    return vel

def update(i):
    # Plots trajectories
    ee_circle = plt.Circle((robot_traj[i, 0], robot_traj[i, 1]), R_EE, fill=False, color='blue', label='desired')
    apf_circle = plt.Circle((robot_pos_apf_history[i, 0], robot_pos_apf_history[i, 1]), R_EE, fill=False,
                            color='cyan', label='apf')
    cbf_circle = plt.Circle((robot_pos_cbf_history[i, 0], robot_pos_cbf_history[i, 1]), R_EE, fill=False,
                            color='black', label='cbf')
    ax[0].add_patch(ee_circle)
    ax[0].add_patch(apf_circle)
    ax[0].add_patch(cbf_circle)
    if not STATIC:
        obs_circle = plt.Circle((obstacle_traj[i, 0], obstacle_traj[i, 1]), R_OBS, fill=False, color='red')
        ax[0].add_patch(obs_circle)

    # Plots APF forces
    ax[1].scatter(i, apf_force_history[i, 0], marker='_', color='red')
    ax[1].scatter(i, apf_force_history[i, 1], marker='_', color='blue')

    if not SAVE_ANIMATION and START < i < END:
        # drawing updated values
        fig.canvas.draw()
        # This will run the GUI event loop until all UI events currently waiting have been processed
        fig.canvas.flush_events()

    if i < N_STEPS-1-SKIP_STEPS:
        ee_circle.remove()
        apf_circle.remove()
        cbf_circle.remove()

        if not STATIC:
            obs_circle.remove()

    return ax

# Create an xy trajectory
robot_traj = traj.ctraj(SE3(0, 0, 0), SE3(1, 0, 0), N_STEPS).t
if STATIC:
    obstacle_traj = STATIC_OBS_POS
else:
    obstacle_traj = traj.ctraj(SE3(0.5, 1, 0), SE3(0.5, -0.5, 0), N_STEPS).t

# Obtain initial robot position for APF and CBF
curr_robot_pos_apf = robot_traj[0, :2]
curr_robot_pos_cbf = robot_traj[0, :2]

# Plotting variables
robot_pos_cbf_history = np.zeros((N_STEPS, 2)); robot_pos_cbf_history[0, :] = robot_traj[0, :2]
robot_pos_apf_history = np.zeros((N_STEPS, 2)); robot_pos_apf_history[0, :] = robot_traj[0, :2]
apf_force_history = np.zeros((N_STEPS, 2))
robot_vel_history = np.zeros((N_STEPS, 2))
robot_vel_apf_history = np.zeros((N_STEPS, 2))
robot_vel_cbf_history = np.zeros((N_STEPS, 2))

for i in range(1, N_STEPS):
    # P controller? incorporates delta time
    if STATIC:
        curr_obstacle_pos = np.array(STATIC_OBS_POS)
    else:
        curr_obstacle_pos = obstacle_traj[i-1, :2]
    curr_robot_pos = robot_traj[i-1, :2]; next_robot_pos = robot_traj[i, :2]
    robot_vel = cap_vel(get_vel(curr_robot_pos, next_robot_pos))

    # APF
    repulsive_force = get_force(curr_robot_pos_apf, curr_obstacle_pos)
    robot_vel_apf = get_vel(curr_robot_pos_apf, next_robot_pos)
    repulsive_vel = cap_vel(robot_vel_apf + repulsive_force)
    robot_x_next_apf = (repulsive_vel * DELTA_TIME) + curr_robot_pos_apf
    curr_robot_pos_apf = robot_x_next_apf

    # CBF
    robot_vel_cbf = use_cbf(curr_robot_pos_cbf, next_robot_pos, curr_obstacle_pos)
    robot_x_next_cbf = curr_robot_pos_cbf + robot_vel_cbf*DELTA_TIME
    curr_robot_pos_cbf = robot_x_next_cbf

    # Save history
    robot_pos_apf_history[i, :] = robot_x_next_apf
    robot_pos_cbf_history[i, :] = robot_x_next_cbf
    apf_force_history[i, :] = repulsive_force
    robot_vel_history[i, :] = robot_vel
    robot_vel_apf_history[i, :] = repulsive_vel
    robot_vel_cbf_history[i, :] = robot_vel_cbf

fig, ax = plt.subplots(2, 1)
vel_fig, vel_ax = plt.subplots(3, 2)
pos_fig, pos_ax = plt.subplots(2, 2)

# Plot velocities
vel_fig.suptitle('Velocity over time')
vel_ax[0][0].plot(range(0, N_STEPS), np.round(robot_vel_history[:, 0], 3), label="x desired velocity", color='red'); vel_ax[0][0].legend()
vel_ax[0][1].plot(range(0, N_STEPS), np.round(robot_vel_history[:, 1], 3), label="y desired velocity", color='red'); vel_ax[0][1].legend()
vel_ax[1][0].plot(range(0, N_STEPS), np.round(robot_vel_apf_history[:, 0], 3), label="x apf velocity", color='blue'); vel_ax[1][0].legend()
vel_ax[1][1].plot(range(0, N_STEPS), np.round(robot_vel_apf_history[:, 1], 3), label="y apf velocity", color='blue'); vel_ax[1][1].legend()
vel_ax[2][0].plot(range(0, N_STEPS), np.round(robot_vel_cbf_history[:, 0], 3), label="x cbf velocity", color='green'); vel_ax[2][0].legend()
vel_ax[2][1].plot(range(0, N_STEPS), np.round(robot_vel_cbf_history[:, 1], 3), label="y cbf velocity", color='green'); vel_ax[2][1].legend()

# Plot position error
pos_fig.suptitle('Position error over time')
pos_ax[0][0].plot(range(0, N_STEPS), np.round(robot_traj[:, 0] - robot_vel_apf_history[:, 0], 3), label="x position error apf", color='red'); pos_ax[0][0].legend()
pos_ax[0][1].plot(range(0, N_STEPS), np.round(robot_traj[:, 1] - robot_vel_apf_history[:, 1], 3), label="y position error apf", color='red'); pos_ax[0][1].legend()
pos_ax[1][0].plot(range(0, N_STEPS), np.round(robot_traj[:, 0] - robot_vel_cbf_history[:, 0], 3), label="x position error cbf", color='blue'); pos_ax[1][0].legend()
pos_ax[1][1].plot(range(0, N_STEPS), np.round(robot_traj[:, 1] - robot_vel_cbf_history[:, 1], 3), label="y position error cbf", color='blue'); pos_ax[1][1].legend()

# Initialise plot
fig.suptitle('Position over time and repulsive forces')
if STATIC:
    obs_circle = plt.Circle((obstacle_traj[0], obstacle_traj[1]), R_OBS, fill=False, color='red')
    ax[0].add_patch(obs_circle)
ax[0].set_xlim([-0.25, 1.5])
ax[0].set_ylim([-0.75, 0.75])

if not SAVE_ANIMATION:
    for i in range(1, N_STEPS - 1, SKIP_STEPS):
        update(i)
else:
    ani = FuncAnimation(fig, update, frames=range(1, N_STEPS - 1, SKIP_STEPS))

ax[0].legend()
ax[1].legend()
if not SAVE_ANIMATION:
    plt.ioff(); plt.legend(), plt.show()
else:
    ani.save(filename=f"{F_NAME}.gif", writer="pillow")
    plt.show()