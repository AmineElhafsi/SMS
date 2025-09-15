"""
# Operational Space Control

Using torque control as well as velocity control to map between the Operational (Task / End-Effector) 
Space and the Joint Space

These controllers are written to be independent of whatever software is used to get the robot's
state / kinematics / dynamics. For instance, we need to know the following details at every timestep:

- Joint positions (and velocities, for torque control)
- End-effector position and orientation
- Dynamics matrices (for torque control), i.e. mass matrix, gravity vector, and Coriolis forces

These values can be obtained from simulation, a real robot, or other robot kinematics + dynamics packages.

For simplicity, we assume that the robot is operating in 3D and that the task is to control the 
position and orientation of the end-effector (6D task). If a different task jacobian is desired, 
this will require some slight modifications to the code (likely, to include a task selection matrix,
which depends on your preferred position/orientation representation).
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike


class OperationalSpaceTorqueController:
    """Operational Space Torque Controller

    Args:
        n_joints (int): Number of joints, e.g. 7 for 7-DOF robot
        kp_task (ArrayLike): Task-space proportional gain(s), shape (6,)
        kd_task (ArrayLike): Task-space derivative gain(s), shape (6,)
        kp_joint (ArrayLike): Joint-space proportional gain(s), shape (n_joints,)
        kd_joint (ArrayLike): Joint-space derivative gain(s), shape (n_joints,)
        tau_min (Optional[ArrayLike]): Minimum joint torques, shape (n_joints,)
        tau_max (Optional[ArrayLike]): Maximum joint torques, shape (n_joints,)
    """

    def __init__(
        self,
        n_joints: int,
        kp_task: ArrayLike,
        kd_task: ArrayLike,
        kp_joint: ArrayLike,
        kd_joint: ArrayLike,
        tau_min: Optional[ArrayLike],
        tau_max: Optional[ArrayLike],
    ):
        self.dim_space = 3  # 3D
        self.dim_task = 6  # Position and orientation in 3D
        assert isinstance(n_joints, int)
        self.n_joints = n_joints
        self.kp_task = _format_gain(kp_task, self.dim_task)
        self.kd_task = _format_gain(kd_task, self.dim_task)
        self.kp_joint = _format_gain(kp_joint, self.n_joints)
        self.kd_joint = _format_gain(kd_joint, self.n_joints)
        self.tau_min = _format_limit(tau_min, self.n_joints, "lower")
        self.tau_max = _format_limit(tau_max, self.n_joints, "upper")

    def __call__(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        pos: np.ndarray,
        rot: np.ndarray,
        des_pos: np.ndarray,
        des_rot: np.ndarray,
        des_vel: Optional[np.ndarray],
        des_omega: Optional[np.ndarray],
        des_accel: Optional[np.ndarray],
        des_alpha: Optional[np.ndarray],
        des_q: Optional[np.ndarray],
        des_qdot: Optional[np.ndarray],
        J: np.ndarray,
        M: np.ndarray,
        M_inv: np.ndarray,
        g: np.ndarray,
        c: np.ndarray,
        check: bool = True,
    ) -> np.ndarray:
        """Compute joint torques for operational space control

        Args:
            q (np.ndarray): Current joint positions, shape (n_joints,)
            qdot (np.ndarray): Current joint velocities, shape (n_joints,)
            pos (np.ndarray): Current end-effector position, shape (3,)
            rot (np.ndarray): Current end-effector rotation matrix, shape (3, 3)
            des_pos (np.ndarray): Desired end-effector position, shape (3,)
            des_rot (np.ndarray): Desired end-effector rotation matrix, shape (3, 3)
            des_vel (Optional[np.ndarray]): Desired end-effector velocity, shape (3,).
                If None, assumed to be zero.
            des_omega (Optional[np.ndarray]): Desired end-effector angular velocity, shape (3,).
                If None, assumed to be zero.
            des_accel (Optional[np.ndarray]): Desired end-effector acceleration, shape (3,).
                If None, assumed to be zero.
            des_alpha (Optional[np.ndarray]): Desired end-effector angular acceleration, shape (3,).
                If None, assumed to be zero.
            des_q (Optional[np.ndarray]): Desired joint positions, shape (n_joints,).
                This is used as a nullspace posture task. If None, assumed to be equal to q.
            des_qdot (Optional[np.ndarray]): Desired joint velocities, shape (n_joints,).
                This is used for a nullspace joint damping term. If None, assumed to be zero.
            J (np.ndarray): Basic Jacobian (mapping joint velocities to task velocities), shape (6, n_joints)
            M (np.ndarray): Mass matrix, shape (n_joints, n_joints)
            M_inv (np.ndarray): Inverse of the mass matrix, shape (n_joints, n_joints)
            g (np.ndarray): Gravity vector, shape (n_joints,)
            c (np.ndarray): Centrifugal/Coriolis vector, shape (n_joints,).
                This term is often neglected, and can be set to a zero vector if so.
            check (bool, optional): Whether to check the shapes of the inputs. Defaults to True.

        Returns:
            np.ndarray: Joint torques, shape (n_joints,)
        """
        if check:
            # Check shapes
            assert q.shape == (self.n_joints,)
            assert qdot.shape == (self.n_joints,)
            assert pos.shape == (self.dim_space,)
            assert rot.shape == (self.dim_space, self.dim_space)
            assert J.shape == (self.dim_task, self.n_joints)
            assert M.shape == (self.n_joints, self.n_joints)
            assert M_inv.shape == (self.n_joints, self.n_joints)
            assert g.shape == (self.n_joints,)
            assert c.shape == (self.n_joints,)
            assert des_pos.shape == (self.dim_space,)
            assert des_rot.shape == (self.dim_space, self.dim_space)
            if des_vel is not None:
                assert des_vel.shape == (self.dim_space,)
            if des_omega is not None:
                assert des_omega.shape == (self.dim_space,)
            if des_accel is not None:
                assert des_accel.shape == (self.dim_space,)
            if des_alpha is not None:
                assert des_alpha.shape == (self.dim_space,)
            if des_q is not None:
                assert des_q.shape == (self.n_joints,)
            if des_qdot is not None:
                assert des_qdot.shape == (self.n_joints,)

        # Assign optional values
        if des_vel is None:
            des_vel = np.zeros(self.dim_space)
        if des_omega is None:
            des_omega = np.zeros(self.dim_space)
        if des_accel is None:
            des_accel = np.zeros(self.dim_space)
        if des_alpha is None:
            des_alpha = np.zeros(self.dim_space)
        if des_q is None:
            des_q = q
        if des_qdot is None:
            des_qdot = np.zeros(self.n_joints)

        # Compute twist
        twist = J @ qdot
        vel = twist[: self.dim_space]
        omega = twist[self.dim_space :]

        # Errors
        pos_error = pos - des_pos
        vel_error = vel - des_vel
        if self.dim_space == 3:
            rot_error = orientation_error_3D(rot, des_rot)
        else:  # self.dim_space == 2:
            rot_error = orientation_error_2D(rot, des_rot)
        omega_error = omega - des_omega
        task_p_error = np.concatenate([pos_error, rot_error])
        task_d_error = np.concatenate([vel_error, omega_error])

        # Operational space matrices
        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = np.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia
        N_bar = np.eye(self.n_joints) - J_bar @ J
        p_bar = J_bar.T @ g

        # Compute task torques
        task_accel = (
            np.concatenate([des_accel, des_alpha])
            - self.kp_task * task_p_error
            - self.kd_task * task_d_error
        )
        task_wrench = task_inertia @ task_accel
        tau = J.T @ (task_wrench + p_bar)

        # Add nullspace joint task
        q_error = q - des_q
        qdot_error = qdot - des_qdot
        joint_accel = -self.kp_joint * q_error - self.kd_joint * qdot_error
        secondary_joint_torques = M @ joint_accel
        tau += N_bar.T @ (secondary_joint_torques + g)

        # Clamp to torque limits
        return np.clip(tau, self.tau_min, self.tau_max)


class OperationalSpaceVelocityController:
    """Operational Space Velocity Controller

    Args:
        n_joints (int): Number of joints, e.g. 7 for 7-DOF robot
        kp_task (ArrayLike): Task-space proportional gain(s), shape (6,)
        kp_joint (ArrayLike): Joint-space proportional gain(s), shape (n_joints,)
        qdot_min (Optional[ArrayLike]): Minimum joint velocities, shape (n_joints,)
        qdot_max (Optional[ArrayLike]): Maximum joint velocities, shape (n_joints,)
    """

    def __init__(
        self,
        n_joints: int,
        kp_task: ArrayLike,
        kp_joint: ArrayLike,
        qdot_min: Optional[ArrayLike],
        qdot_max: Optional[ArrayLike],
    ):
        self.dim_space = 3  # 3D
        self.dim_task = 6  # Position and orientation in 3D
        assert isinstance(n_joints, int)
        self.n_joints = n_joints
        self.kp_task = _format_gain(kp_task, self.dim_task)
        self.kp_joint = _format_gain(kp_joint, self.n_joints)
        self.qdot_min = _format_limit(qdot_min, self.n_joints, "lower")
        self.qdot_max = _format_limit(qdot_max, self.n_joints, "upper")

    def __call__(
        self,
        q: np.ndarray,
        pos: np.ndarray,
        rot: np.ndarray,
        des_pos: np.ndarray,
        des_rot: np.ndarray,
        des_vel: Optional[np.ndarray],
        des_omega: Optional[np.ndarray],
        des_q: Optional[np.ndarray],
        J: np.ndarray,
        check: bool = True,
    ):
        """Compute joint velocities for operational space control

        Args:
            q (np.ndarray): Current joint positions, shape (n_joints,)
            pos (np.ndarray): Current end-effector position, shape (3,)
            rot (np.ndarray): Current end-effector rotation matrix, shape (3, 3)
            des_pos (np.ndarray): Desired end-effector position, shape (3,)
            des_rot (np.ndarray): Desired end-effector rotation matrix, shape (3, 3)
            des_vel (Optional[np.ndarray]): Desired end-effector velocity, shape (3,).
                If None, assumed to be zero.
            des_omega (Optional[np.ndarray]): Desired end-effector angular velocity, shape (3,).
                If None, assumed to be zero.
            des_q (Optional[np.ndarray]): Desired joint positions, shape (n_joints,).
                This is used as a nullspace posture task. If None, assumed to be equal to q.
            J (np.ndarray): Basic Jacobian (mapping joint velocities to task velocities), shape (6, n_joints)
            check (bool, optional): Whether to check the shapes of the inputs. Defaults to True.

        Returns:
            np.ndarray: Joint torques, shape (n_joints,)
        """
        if check:
            # Check shapes
            assert q.shape == (self.n_joints,)
            assert pos.shape == (self.dim_space,)
            assert rot.shape == (self.dim_space, self.dim_space)
            assert J.shape == (self.dim_task, self.n_joints)
            if des_pos is not None:
                assert des_pos.shape == (self.dim_space,)
            if des_rot is not None:
                assert des_rot.shape == (self.dim_space, self.dim_space)
            if des_vel is not None:
                assert des_vel.shape == (self.dim_space,)
            if des_omega is not None:
                assert des_omega.shape == (self.dim_space,)
            if des_q is not None:
                assert des_q.shape == (self.n_joints,)

        # Assign optional values
        if des_vel is None:
            des_vel = np.zeros(self.dim_space)
        if des_omega is None:
            des_omega = np.zeros(self.dim_space)
        if des_q is None:
            des_q = q

        # Errors
        pos_error = pos - des_pos
        if self.dim_space == 3:
            rot_error = orientation_error_3D(rot, des_rot)
        else:  # self.dim_space == 2:
            rot_error = orientation_error_2D(rot, des_rot)
        task_p_error = np.concatenate([pos_error, rot_error])

        J_pseudo = np.linalg.pinv(J)
        N = np.eye(self.n_joints) - J_pseudo @ J

        # Compute task velocities
        task_vel = np.concatenate([des_vel, des_omega]) - self.kp_task * task_p_error
        # Map to joint velocities
        v = J_pseudo @ task_vel
        # Add nullspace joint task
        q_error = q - des_q
        secondary_joint_vel = -self.kp_joint * q_error
        v += N.T @ secondary_joint_vel

        # Clamp to velocity limits
        return np.clip(v, self.qdot_min, self.qdot_max)


# Helper functions


def orientation_error_3D(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Determine the angular error vector between two rotation matrices in 3D.

    Args:
        R_cur (np.ndarray): Current rotation matrix, shape (3, 3)
        R_des (np.ndarray): Desired rotation matrix, shape (3, 3)

    Returns:
        np.ndarray: Angular error, shape (3,)
    """
    return -0.5 * (
        np.cross(R_cur[:, 0], R_des[:, 0])
        + np.cross(R_cur[:, 1], R_des[:, 1])
        + np.cross(R_cur[:, 2], R_des[:, 2])
    )


def orientation_error_2D(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Determine the angular error vector between two rotation matrices in 2D.

    Args:
        R_cur (np.ndarray): Current rotation matrix, shape (2, 2)
        R_des (np.ndarray): Desired rotation matrix, shape (2, 2)

    Returns:
        np.ndarray: Angular error, shape (1,)
    """
    return -0.5 * (
        np.cross(R_cur[:, 0], R_des[:, 0]) + np.cross(R_cur[:, 1], R_des[:, 1])
    )


def _format_gain(k: float, dim: int) -> np.ndarray:
    if isinstance(k, (float, int)):
        k = k * np.ones(dim)
    else:
        k = np.atleast_1d(k).flatten()
        assert k.shape == (dim,)
    return k


def _format_limit(
    arr: Optional[ArrayLike], dim: int, side: str
) -> Optional[np.ndarray]:
    if arr is None:
        if side == "upper":
            arr = np.inf * np.ones(dim)
        elif side == "lower":
            arr = -np.inf * np.ones(dim)
        else:
            raise ValueError(f"Invalid side: {side}")
    else:
        arr = np.atleast_1d(arr).flatten()
        assert arr.shape == (dim,)
    return arr
