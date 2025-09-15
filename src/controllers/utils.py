from typing import Dict

import numpy as np

from src.opspace.trajectories.bezier import bezier_trajectory
from src.opspace.trajectories.splines import CompositeBezierCurve

def generate_strike_reference_motion(
    contact_position: np.ndarray,
    contact_speed: float, 
    contact_angle: float,
    dt: float,
    l_accel: float = 0.1,
    l_decel: float = 0.1,
) -> Dict:
    """
    Generate a reference motion for a manipulator strike action. The 
    end effector will move to the contact position with the specified
    speed and angle. The motion will be split into three phases:
    1. Preparation Phase: Move to the start position of the strike motion
    2. Acceleration Phase: Move to the contact position with a constant speed
    3. Deceleration Phase: Move to the end position of the strike motion while
        decelerating to a stop.
    
    Args:
        contact_position (np.ndarray): The position of the contact point in the ROBOT frame
        contact_speed (float): The speed of the end effector at the contact point
        contact_angle (float): The angle of the paddle at the contact point
        l_accel (float): The distance to accelerate over
        l_decel (float): The distance to decelerate over
    """
    # account for robot position offset
    # contact_position = contact_position - self.robot.get_pos().cpu().numpy()[0]

    # get path start and end points
    direction_vector = np.array(
        [np.cos(contact_angle), np.sin(contact_angle), 0]
    )
    start_position = contact_position - l_accel * direction_vector
    end_position = contact_position + l_decel * direction_vector

    # get contact velocity vector
    contact_velocity = contact_speed * direction_vector

    # phase timing estimates
    t_prep = 2.0
    t_contact = (l_accel / contact_speed) + t_prep
    t_end = (l_decel / contact_speed) + t_contact

    # assemble waypoints (set path constraints)
    waypoint_pos = {
        "reset": [start_position[0], start_position[1], 0.4],
        "accel_phase": start_position,
        "contact": contact_position,
        "decel_phase": end_position,
    }
    waypoint_vel = {
        "reset": [0, 0, 0],
        "accel_phase": np.array([0., 0., 0.]),
        "contact": contact_velocity,
        "decel_phase": np.array([0., 0., 0.]),
    }
    waypoint_times = {
        "reset": 0,
        "accel_phase": t_prep,
        "contact": t_contact,
        "decel_phase": t_end,
    }

    # generate bezier curves
    prep_curve, _ = bezier_trajectory(
        p0=waypoint_pos["reset"],
        pf=waypoint_pos["accel_phase"],
        t0=waypoint_times["reset"],
        tf=waypoint_times["accel_phase"],
        n_control_pts=10,
        v0=waypoint_vel["reset"],
        vf=waypoint_vel["accel_phase"],
    )
    accel_curve, _ = bezier_trajectory(
        p0=waypoint_pos["accel_phase"],
        pf=waypoint_pos["contact"],
        t0=waypoint_times["accel_phase"],
        tf=waypoint_times["contact"],
        n_control_pts=10,
        v0=waypoint_vel["accel_phase"],
        vf=waypoint_vel["contact"],
    )
    decel_curve, _ = bezier_trajectory(
        p0=waypoint_pos["contact"],
        pf=waypoint_pos["decel_phase"],
        t0=waypoint_times["contact"],
        tf=waypoint_times["decel_phase"],
        n_control_pts=10,
        v0=waypoint_vel["contact"],
        vf=waypoint_vel["decel_phase"],
    )
    curve = CompositeBezierCurve([prep_curve, accel_curve, decel_curve])

    times = np.arange(curve.a, curve.b + dt, dt)
    times[-1] = curve.b # fix floating point issue sometimes

    reference_motion = {
        "times": times,
        "positions": curve(times),
        "velocities": curve.derivative(times),
        "accelerations": curve.derivative.derivative(times),
        "paddle_angle": contact_angle,

    }

    #######
    # import matplotlib.pyplot as plt

    # # Normalize the time values to range between 0 and 1
    # norm = plt.Normalize(times.min(), times.max())

    # # Create a colormap
    # cmap = plt.get_cmap('viridis')

    # # Plot the trajectory in 3D with a colormap
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot each segment of the trajectory with a color based on the time
    # # for i in range(len(times) - 1):
    # #     ax.plot(self.positions[i:i+2, 0], self.positions[i:i+2, 1], self.positions[i:i+2, 2], color=cmap(norm(times[i])))

    # # plot trajectory as scatter plot, color-coded by segment
    # ax.scatter(reference_motion["positions"][:, 0], reference_motion["positions"][:, 1], reference_motion["positions"][:, 2], c=times, cmap=cmap, s = 0.02)

    # # plot start, contact, and end points
    # ax.scatter(waypoint_pos["reset"][0], waypoint_pos["reset"][1], waypoint_pos["reset"][2], c='r', s=10)
    # ax.scatter(waypoint_pos["accel_phase"][0], waypoint_pos["accel_phase"][1], waypoint_pos["accel_phase"][2], c='g', s=10)
    # ax.scatter(waypoint_pos["contact"][0], waypoint_pos["contact"][1], waypoint_pos["contact"][2], c='b', s=10)
    # ax.scatter(waypoint_pos["decel_phase"][0], waypoint_pos["decel_phase"][1], waypoint_pos["decel_phase"][2], c='y', s=10)

    # # show contact point as a large red sphere
    # ax.scatter(contact_position[0], contact_position[1], contact_position[2], c='r', s=15)

    # # Add a colorbar
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # fig.colorbar(sm, ax=ax, label='Time (s)')

    # # make axes equal with xlim (-1, 1), ylim (-1, 1), zlim (0, 1)
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(0, 1)

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # plt.show()
    #######

    return reference_motion