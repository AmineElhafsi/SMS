"""Example of using operational space control to hold a desired end-effector pose

Since this uses torque control, the robot is compliant. Interacting with it in the Pybullet
GUI will allow it to be displaced and 'snap back' to the desired pose
"""

import numpy as np
import pinocchio as pin
import pybullet
import pybullet_data

from opspace.utils.controllers import OperationalSpaceTorqueController


def main():
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

    kp_pos = 5.0
    kp_rot = 2.0
    kd_pos = 2.0
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
            des_pos=target_pos,
            des_rot=target_rmat,
            des_vel=np.zeros(3),
            des_omega=np.zeros(3),
            des_accel=np.zeros(3),
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


if __name__ == "__main__":
    main()
