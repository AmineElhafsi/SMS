"""
Example of using operational space control to follow a trajectory, 
where that trajectory is computed to achieve a desired contact position and velocity

The trajectory is composed of multiple chained bezier curves with boundary conditions
on position and velocity, for given knot times. Each curve individually minimizes jerk.

It's possible to do a more complex optimization over a full spline, with knot-point
retiming and a free-final-time formulation. But, this might be overkill for this case
"""

import numpy as np
import pinocchio as pin
import pybullet
import pybullet_data
import time
from opspace.utils.controllers import OperationalSpaceTorqueController
from opspace.trajectories.bezier import bezier_trajectory
from opspace.trajectories.splines import CompositeBezierCurve

table_height = 0.25

waypoint_pos = {
    "reset": [0.3, 0, 0.4],
    "start": [0.3, 0, table_height],
    "contact": [0.5, 0, table_height],
    "halt": [0.6, 0, table_height],
    "return": [0.3, 0, 0.4],
}
waypoint_vel = {
    "reset": [0, 0, 0],
    "start": [0, 0, 0],
    "contact": [0.2, 0, 0],
    "halt": [0, 0, 0],
    "return": [0, 0, 0],
}
waypoint_times = {
    "reset": 0,
    "start": 2.0,
    "contact": 3.0,
    "halt": 4.0,
    "return": 6.0,
}

curve_1, _ = bezier_trajectory(
    p0=waypoint_pos["reset"],
    pf=waypoint_pos["start"],
    t0=waypoint_times["reset"],
    tf=waypoint_times["start"],
    n_control_pts=10,
    v0=waypoint_vel["reset"],
    vf=waypoint_vel["start"],
)
curve_2, _ = bezier_trajectory(
    p0=waypoint_pos["start"],
    pf=waypoint_pos["contact"],
    t0=waypoint_times["start"],
    tf=waypoint_times["contact"],
    n_control_pts=10,
    v0=waypoint_vel["start"],
    vf=waypoint_vel["contact"],
)
curve_3, _ = bezier_trajectory(
    p0=waypoint_pos["contact"],
    pf=waypoint_pos["halt"],
    t0=waypoint_times["contact"],
    tf=waypoint_times["halt"],
    n_control_pts=10,
    v0=waypoint_vel["contact"],
    vf=waypoint_vel["halt"],
)
curve_4, _ = bezier_trajectory(
    p0=waypoint_pos["halt"],
    pf=waypoint_pos["return"],
    t0=waypoint_times["halt"],
    tf=waypoint_times["return"],
    n_control_pts=10,
    v0=waypoint_vel["halt"],
    vf=waypoint_vel["return"],
)
curve = CompositeBezierCurve([curve_1, curve_2, curve_3, curve_4])
dt = 1 / 240
times = np.arange(curve.a, curve.b + dt, dt)
times[-1] = curve.b  # Fix floating point issue sometimes
positions = curve(times)
velocities = curve.derivative(times)
accelerations = curve.derivative.derivative(times)
num_timesteps = len(times)


# Set up simulation
pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
pybullet.setGravity(0, 0, -9.81)
pybullet.loadURDF("plane.urdf")
robot_id = pybullet.loadURDF(
    "opspace/assets/franka_panda/panda.urdf",
    [0, 0, 0],
    useFixedBase=True,
    flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
)
# "Unlock" the default controller
for i in range(7):
    pybullet.setJointMotorControl2(robot_id, i, pybullet.VELOCITY_CONTROL, force=0)

# Set up Pinocchio model
model = pin.buildModelFromUrdf("opspace/assets/franka_panda/panda.urdf")
data = pin.Data(model)

# Initial configuration
initial_joint_positions = np.array(
    [
        0.0,
        -np.pi / 6,
        0.0,
        -3 * np.pi / 4,
        0.0,
        5 * np.pi / 9,
        0.0,
    ]
)

# Set initial joint positions in PyBullet
for i, pos in enumerate(initial_joint_positions):
    pybullet.resetJointState(robot_id, i, pos)

# Target end-effector position
target_pos = np.array([0.374903, -5.69483e-12, 0.409353])
target_rmat = np.array(
    [
        [0.704416, 0.704416, -0.0871557],
        [0.707107, -0.707107, -8.04082e-12],
        [-0.0616284, -0.0616284, -0.996195],
    ]
)

kp_pos = 50.0
kp_rot = 2.0
kd_pos = 20.0
kd_rot = 1.0
controller = OperationalSpaceTorqueController(
    n_joints=7,
    kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
    kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
    kp_joint=1.0,
    kd_joint=0.5,
    tau_min=None,
    tau_max=None,
)

# Simulation loop
while True:
    for i in range(num_timesteps):
        # Get current joint states
        joint_states = pybullet.getJointStates(robot_id, range(7))
        q = np.array([state[0] for state in joint_states])
        dq = np.array([state[1] for state in joint_states])

        pin.forwardKinematics(model, data, q, dq)
        pin.updateFramePlacements(model, data)
        ee_frame_id = model.getFrameId("panda_grasptarget_hand")
        current_ee_state = data.oMf[ee_frame_id]
        current_pos = current_ee_state.translation
        current_rmat = current_ee_state.rotation
        J = pin.computeFrameJacobian(
            model, data, q, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        M = pin.crba(model, data, q)
        M_inv = np.linalg.inv(M)
        bias = pin.nle(model, data, q, dq)

        tau = controller(
            q=q,
            qdot=dq,
            pos=current_pos,
            rot=current_rmat,
            des_pos=positions[i],
            des_rot=target_rmat,
            des_vel=velocities[i],
            des_omega=np.zeros(3),
            des_accel=accelerations[i],
            des_alpha=np.zeros(3),
            des_q=initial_joint_positions,
            des_qdot=np.zeros(7),
            J=J,
            M=M,
            M_inv=M_inv,
            g=bias,
            c=np.zeros(7),
            check=True,
        )

        pybullet.setJointMotorControlArray(
            robot_id, list(range(7)), pybullet.TORQUE_CONTROL, forces=tau
        )
        pybullet.stepSimulation()
        time.sleep(1 / 500)
