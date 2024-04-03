"""
Compares CBF, APF and no OA in a situation where the obstacle is static and dynamic
Does not take into account the area of the agents and obstacles (ie only point to point distance)
"""
import sys
import time

import matplotlib
matplotlib.use('GTK3Agg')  # Trying to make matplotlib faster
matplotlib.interactive(True)  # Needed for 3D plot updates

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from roboticstoolbox.tools import trajectory as traj
from spatialmath import SE3
from qpsolvers import solve_ls
import superquadrics as sq
import volumetric as vd
import math
import numpy as np

# Plots
SAVE_ANIMATION = False  # If you want to save an animation
F_NAME = "static"  # Name of the gif animation
SKIP_STEPS = 1  # Useful for skipping steps during simulation to increase speed

# STATIC = False; START = 230; END = 500
STATIC = True; START = 100; END = 500; STATIC_OBS_POS = (0.5, 0.1, 0.01)  # 115 start and (0.3, 0.1) obs pos is default

# CBF param
CBF_GAMMA = 2  # Represents how close the agent can come to the obstacle (higher = closer)

# APF params
APF_Q = 0.2  # Minimum allowable distance before the potential field is activated
NETA = 0.5  # Gain for the repulsive field

# P controller used in cbf and apf
KV = 1  # Velocity gain used when calculating velocity
MAX_VEL = 1.5  # Maximum allowable velocity
MIN_VEL = -1.5  # Minimum allowable velocity

TIME = 1  # Desired time to complete the task?

########################################################################################################################
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))
ax.view_init(0, -0)

if not SAVE_ANIMATION:
    plt.ion()

FREQ = 500  # Control frequency of the universal robot E series is 500Hz
DELTA_TIME = 1/FREQ  # Time period
N_STEPS = int(TIME/DELTA_TIME)  # Total number of steps for the simulation

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
    curr_dist = vol_dist_calcs.get_h()
    if curr_dist < APF_Q:
        nabla_d = vol_dist_calcs.get_h_dot().flatten()
        force = -NETA * (1/APF_Q - 1/curr_dist) * (1/curr_dist)**2 * nabla_d
    else:
        force = np.array([0, 0, 0])

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
    R = np.identity(3)  # Weights?
    s = KV*((desired_pos - r) / DELTA_TIME)  # What we are trying to optimise
    G = -vol_dist_calcs.get_h_dot()  # CBF derivative
    h = np.array([CBF_GAMMA * vol_dist_calcs.get_h()])  # CBF exponential gamma*hx
    vel = solve_ls(R, s, G, h, lb=np.array([MIN_VEL, MIN_VEL, MIN_VEL]), ub=np.array([MAX_VEL, MAX_VEL, MAX_VEL]), solver="cvxopt")  # osqp or clarabel or cvxopt or quadprog
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

# Create an xy trajectory
robot_traj = traj.ctraj(SE3(0, -0.1, -0.01), SE3(1, -0.1, 0.0), N_STEPS).t
if STATIC:
    obstacle_traj = np.full(robot_traj.shape, STATIC_OBS_POS)
else:
    obstacle_traj = traj.ctraj(SE3(0.5, 0.5, 0), SE3(0.5, -0.5, 0.01), N_STEPS).t

# Obtain initial robot position for APF and CBF
curr_robot_pos_apf = robot_traj[0]
curr_robot_pos_cbf = robot_traj[0]

# Plotting variables
robot_pos_cbf_history = np.zeros((N_STEPS, 3)); robot_pos_cbf_history[0, :] = robot_traj[0]
robot_pos_apf_history = np.zeros((N_STEPS, 3)); robot_pos_apf_history[0, :] = robot_traj[0]
apf_force_history = np.zeros((N_STEPS, 3))
cbf_h_history = np.zeros((N_STEPS, 1))
robot_vel_history = np.zeros((N_STEPS, 3))
robot_vel_apf_history = np.zeros((N_STEPS, 3))
robot_vel_cbf_history = np.zeros((N_STEPS, 3))

ee_sq = sq.SuperquadricObject(a=0.22, b=0.21, c=0.05, eps1=1.0, eps2=1.0, pose=(robot_traj[0], np.array([1, 0, 0, 0])))
obs_sq = sq.SuperquadricObject(a=0.11, b=0.13, c=0.03, eps1=0.1, eps2=1.0, pose=(obstacle_traj[0], np.array([1, 0, 0, 0])))
ee_sq.plot_sq(ax, 'blue', plot_type='3D', alpha=0.2)
obs_sq.plot_sq(ax, 'red', plot_type='3D', alpha=0.2)

vol_dist_calcs = vd.VolumetricDistance(ee_sq, obs_sq)

ax.set_xlim([-0.25, 1.5])
ax.set_ylim([-0.75, 0.75])

for i in range(1, N_STEPS):
    # TODO: Update position of the superquadrics
    # P controller? incorporates delta time
    curr_obstacle_pos = obstacle_traj[i-1]; next_obstacle_pos = obstacle_traj[i]
    curr_robot_pos = robot_traj[i-1]; next_robot_pos = robot_traj[i]
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
    cbf_h_history[i, 0] = vol_dist_calcs.get_h()
    ee_sq.update_scene(robot_x_next_cbf)

    # Save history
    robot_pos_apf_history[i, :] = robot_x_next_apf
    robot_pos_cbf_history[i, :] = robot_x_next_cbf
    apf_force_history[i, :] = repulsive_force
    robot_vel_history[i, :] = robot_vel
    robot_vel_apf_history[i, :] = repulsive_vel
    robot_vel_cbf_history[i, :] = robot_vel_cbf

# Plot velocities
vel_fig, vel_ax = plt.subplots(3, 3)
vel_fig.suptitle('Velocity over time')
# vel_ax[0][0].plot(range(0, N_STEPS), np.round(robot_vel_history[:, 0], 3), label="x desired velocity", color='red'); vel_ax[0][0].legend()
# vel_ax[0][1].plot(range(0, N_STEPS), np.round(robot_vel_history[:, 1], 3), label="y desired velocity", color='red'); vel_ax[0][1].legend()
# vel_ax[1][0].plot(range(0, N_STEPS), np.round(robot_vel_apf_history[:, 0], 3), label="x apf velocity", color='blue'); vel_ax[1][0].legend()
# vel_ax[1][1].plot(range(0, N_STEPS), np.round(robot_vel_apf_history[:, 1], 3), label="y apf velocity", color='blue'); vel_ax[1][1].legend()
vel_ax[2][0].plot(range(0, N_STEPS), np.round(robot_vel_cbf_history[:, 0], 3), label="x cbf velocity", color='green'); vel_ax[2][0].legend()
vel_ax[2][1].plot(range(0, N_STEPS), np.round(robot_vel_cbf_history[:, 1], 3), label="y cbf velocity", color='green'); vel_ax[2][1].legend()
vel_ax[2][2].plot(range(0, N_STEPS), np.round(robot_vel_cbf_history[:, 2], 3), label="z cbf velocity", color='green'); vel_ax[2][1].legend()

# Plot position error
pos_fig, pos_ax = plt.subplots(2, 3)
pos_fig.suptitle('Position error over time')
# pos_ax[0][0].plot(range(0, N_STEPS), np.round(robot_traj[:, 0] - robot_pos_apf_history[:, 0], 3), label="x position error apf", color='red'); pos_ax[0][0].legend()
# pos_ax[0][1].plot(range(0, N_STEPS), np.round(robot_traj[:, 1] - robot_pos_apf_history[:, 1], 3), label="y position error apf", color='red'); pos_ax[0][1].legend()
# pos_ax[0][2].plot(range(0, N_STEPS), np.round(robot_traj[:, 2] - robot_pos_apf_history[:, 2], 3), label="z position error apf", color='red'); pos_ax[0][1].legend()
pos_ax[1][0].plot(range(0, N_STEPS), np.round(robot_traj[:, 0] - robot_pos_cbf_history[:, 0], 3), label="x position error cbf", color='blue'); pos_ax[1][0].legend()
pos_ax[1][1].plot(range(0, N_STEPS), np.round(robot_traj[:, 1] - robot_pos_cbf_history[:, 1], 3), label="y position error cbf", color='blue'); pos_ax[1][1].legend()
pos_ax[1][2].plot(range(0, N_STEPS), np.round(robot_traj[:, 2] - robot_pos_cbf_history[:, 2], 3), label="z position error cbf", color='blue'); pos_ax[1][2].legend()

# Initialise plot
fig.suptitle('Position over time and repulsive forces')
ax.scatter(robot_traj[0, 0], robot_traj[0, 1], robot_traj[0, 2], marker='.', color='blue', label='desired')
# ax.scatter(robot_pos_apf_history[0, 0], robot_pos_apf_history[0, 1], robot_pos_apf_history[0, 2], marker='.', color='cyan', label='apf')
ax.scatter(robot_pos_cbf_history[0, 0], robot_pos_cbf_history[0, 1], robot_pos_cbf_history[0, 2], marker='.', color='black', label='cbf')

fig_params, ax_params = plt.subplots(1, 2)
ax_params[0].scatter(0, apf_force_history[0, 0], marker='_', color='red', label="x repulsion")
ax_params[0].scatter(0, apf_force_history[0, 1], marker='_', color='blue', label="y repulsion")
ax_params[0].scatter(0, apf_force_history[0, 2], marker='_', color='green', label="y repulsion")
ax_params[1].scatter(0, cbf_h_history[0, 0], marker='.', color='red', label="x repulsion")


def update(i):
    # Plots trajectories
    ax.scatter(robot_traj[i, 0], robot_traj[i, 1], robot_traj[i, 2], marker='.', color='blue')
    # ax.scatter(robot_pos_apf_history[i, 0], robot_pos_apf_history[i, 1], robot_pos_apf_history[0, 2], marker='.', color='cyan')
    ax.scatter(robot_pos_cbf_history[i, 0], robot_pos_cbf_history[i, 1], robot_pos_cbf_history[i, 2], marker='.', color='black')
    if STATIC:
        ax.scatter(STATIC_OBS_POS[0], STATIC_OBS_POS[1], marker='*', color='red')
    else:
        ax.scatter(obstacle_traj[i, 0], obstacle_traj[i, 1], marker='*', color='red')
    ee_sq.update_scene(robot_pos_cbf_history[i])

    # Plots APF forces and h value
    # ax_params[1].scatter(i, apf_force_history[i, 0], marker='_', color='red')
    # ax_params[1].scatter(i, apf_force_history[i, 1], marker='_', color='blue')
    # ax_params[0].scatter(i, apf_force_history[0, 2], marker='_', color='green', label="z repulsion")
    ax_params[1].scatter(i, cbf_h_history[i, 0], marker='.', color='red', label="x repulsion")

    if START < i < END and not SAVE_ANIMATION:
        if vol_dist_calcs.get_h() < 0.5:
            ass = ee_sq.plot_sq(ax, 'blue')
            time.sleep(0.01)
        # drawing updated values
        fig.canvas.draw()
        # This will run the GUI event loop until all UI events currently waiting have been processed
        fig.canvas.flush_events()
        if vol_dist_calcs.get_h() < 0.5:
            ass.remove()
    return ax


if not SAVE_ANIMATION:
    for i in range(1, N_STEPS - 1, SKIP_STEPS):
        update(i)
else:
    ani = FuncAnimation(fig, update, frames=range(1, N_STEPS - 1, SKIP_STEPS))
# plt.show()
if not SAVE_ANIMATION:
    plt.ioff(); plt.legend(), plt.show()
# else:
#     ani.save(filename=f"{F_NAME}.gif", writer="pillow")
#     plt.show()