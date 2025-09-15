from typing import Dict, List

import numpy as np
import pinocchio as pin

from src.opspace.utils.controllers import OperationalSpaceTorqueController, OperationalSpaceVelocityController

class BilliardsManipulatorController:
    def __init__(
        self,
        robot_urdf_path: str,
        joint_names: List[str],
        control_dt,
        kp_pos: float = 1000.0, # 100.0, # 50.0,
        kp_rot: float = 300.0, #7.5, # 2.0,
        kd_pos: float = 800.0, # 45.0, # 20.0,
        kd_rot: float = 5.0, #1.0,
        kp_joint: float = 200.0, #3.0, #1.0,
        kd_joint: float = 4.5, #0.5,
    ):
        # build robot model
        self.robot_urdf_path = robot_urdf_path
        self.model = pin.buildModelFromUrdf(robot_urdf_path)

        # lock uncontrolled joints
        lock_ids = []
        for name in self.model.names:
            if name not in joint_names and name != "universe":
                lock_ids.append(self.model.getJointId(name))
                print(f"Locking joint {name}.")
        initial_config = np.zeros(self.model.nq)
        self.model = pin.buildReducedModel(self.model, lock_ids, initial_config)
        self.data = pin.Data(self.model)

        # control configuration
        self.control_dt = control_dt
        self.n_joints = len(joint_names)
        self.kp_pos = kp_pos
        self.kp_rot = kp_rot
        self.kd_pos = kd_pos
        self.kd_rot = kd_rot
        self.kp_joint = kp_joint
        self.kd_joint = kd_joint

        self.controller = OperationalSpaceTorqueController(
            n_joints=self.n_joints,
            kp_task=np.concatenate([self.kp_pos * np.ones(3), self.kp_rot * np.ones(3)]),
            kd_task=np.concatenate([self.kd_pos * np.ones(3), self.kd_rot * np.ones(3)]),
            kp_joint=self.kp_joint,
            kd_joint=self.kd_joint,
            tau_min=np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0]),
            tau_max=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]),
        )

        # self.controller = OperationalSpaceVelocityController(
        #     n_joints=self.n_joints,
        #     kp_task=np.concatenate([self.kp_pos * np.ones(3), self.kp_rot * np.ones(3)]),
        #     kp_joint=self.kp_joint,
        #     qdot_min=np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]),
        #     qdot_max=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        #     # qdot_min=np.array([-2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100]),
        #     # qdot_max=np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),
        # )

        # initial robot configuration
        self.initial_joint_positions = np.array(
            [
                0.0,
                -np.pi / 6,
                0.0,
                -3 * np.pi / 4,
                0.0,
                5 * np.pi / 9,
                - np.pi / 4.0,
            ]
        )
        
        # trajectory information
        self.times = None
        self.positions = None
        self.velocities = None
        self.accelerations = None
        self.num_timesteps = None
        self.paddle_angle = None

        # tracking information
        self.current_timestep = 0

    def reset(self):
        # trajectory information
        self.times = None
        self.positions = None
        self.velocities = None
        self.accelerations = None
        self.num_timesteps = None
        self.paddle_angle = None

        # tracking information
        self.current_timestep = 0

    def set_reference_motion(self, reference_motion: Dict):
        # assert trajectory information is empty (otherwise indicate that trajectory info already exists)
        assert self.times is None \
            and self.positions is None \
            and self.velocities is None \
            and self.accelerations is None \
            and self.paddle_angle is None, \
            "Trajectory information already exists, call controller.reset()"

        self.times = reference_motion["times"]
        self.positions = reference_motion["positions"]
        self.velocities = reference_motion["velocities"]
        self.accelerations = reference_motion["accelerations"]
        self.num_timesteps = len(self.times)
        self.paddle_angle = reference_motion["paddle_angle"]

    def get_command(
        self,
        current_q: np.ndarray,
        current_dq: np.ndarray,
    ):
        # check if trajectory is generated
        if self.times is None:
            raise ValueError("Trajectory not generated yet")
        
        # FK and update frame placements
        pin.forwardKinematics(self.model, self.data, current_q, current_dq)
        pin.updateFramePlacements(self.model, self.data)

        # get current end-effector status
        ee_frame_id = self.model.getFrameId("panda_grasptarget_hand")
        current_ee_state = self.data.oMf[ee_frame_id]
        current_pos = current_ee_state.translation
        current_rmat = current_ee_state.rotation
        J = pin.computeFrameJacobian(
            self.model, self.data, current_q, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # update
        M = pin.crba(self.model, self.data, current_q)
        M_inv = np.linalg.inv(M)
        bias = pin.nle(self.model, self.data, current_q, current_dq)

        # rotation matrix about z axis
        target_rmat = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        target_rmat = np.array(
            [
                [np.cos(self.paddle_angle), -np.sin(self.paddle_angle), 0.0],
                [np.sin(self.paddle_angle), np.cos(self.paddle_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ) @ target_rmat


        # get control command
        tau = self.controller(
            q=current_q,
            qdot=current_dq,
            pos=current_pos,
            rot=current_rmat,
            des_pos=self.positions[self.current_timestep],
            des_rot=target_rmat,
            des_vel=self.velocities[self.current_timestep],
            des_omega=np.zeros(3),
            des_accel=self.accelerations[self.current_timestep],
            des_alpha=np.zeros(3),
            des_q=self.initial_joint_positions,
            des_qdot=np.zeros(7),
            J=J,
            M=M,
            M_inv=M_inv,
            g=bias,
            c=np.zeros(7),
            check=True,
        )

        # v_command = self.controller(
        #     q=current_q,
        #     pos=current_pos,
        #     rot=current_rmat,
        #     des_pos=self.positions[self.current_timestep],
        #     des_rot=target_rmat,
        #     des_vel=self.velocities[self.current_timestep],
        #     des_omega=np.zeros(3),
        #     des_q=self.initial_joint_positions,
        #     J=J,
        #     check=True,
        # )

        if self.current_timestep < self.num_timesteps - 1:
            self.current_timestep += 1

        # return v_command
        return tau



        




