from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pybullet as p 
import yaml

from src.controllers.manipulator import BilliardsManipulatorController
from src.physics.pybullet.base import BaseEnv
from src.physics.sim_entities import load_sim_object, Sphere, Mesh
from src.opspace.trajectories.bezier import bezier_trajectory
from src.opspace.trajectories.splines import CompositeBezierCurve
from src.optimization.nelder_mead import nelder_mead
from src.utils.io import create_directory, get_unique_file_path, save_dict_to_json


class Billiards(BaseEnv):
    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__("billiards", config)

        # prepare data logging
        self.target_ball_trajectory = []
        self.robot_joint_trajectory = []
            
    def parse_task(self) -> Dict:
        # load task config
        scenario_config_path = Path(self.config["dataset_config"]["dataset_path"]) / "scenario_config.yaml"
        with open(scenario_config_path, "r") as file:
            scenario_config = yaml.safe_load(file)
        task_config = scenario_config["task"]

        # assemble list of balls among scene entities
        billiard_balls = []
        for entity_name, entity_id in self.sim_entities.items():
            collision_shape_data = p.getCollisionShapeData(entity_id, -1)
            geometry_type = collision_shape_data[0][2]
            if geometry_type == p.GEOM_SPHERE:
                ball_pos = np.array(p.getBasePositionAndOrientation(entity_id)[0])
                billiard_balls.append((entity_name, ball_pos))

        # identify cue ball
        cue_ball_pos = np.array(task_config["cue_ball_position"])
        cue_ball_name, cue_ball_pos = min(billiard_balls, key=lambda x: np.linalg.norm(x[1] - cue_ball_pos))

        # identify target ball
        target_ball_pos = np.array(task_config["target_ball_position"])
        target_ball_name, target_ball_pos = min(billiard_balls, key=lambda x: np.linalg.norm(x[1] - target_ball_pos))

        # define task-relevant variables
        self.cue_ball_id = self.sim_entities[cue_ball_name]
        self.target_ball_id = self.sim_entities[target_ball_name]
        self.goal_position = np.array(task_config["goal_position"])

        # add visual for goal
        # visual_shape_id = p.createVisualShape(
        #         shapeType=p.GEOM_CYLINDER,
        #         radius=0.05,
        #         length=0.01,  # very thin to make it look like a flat circle
        #         rgbaColor=[0, 1, 0, 0.5],
        #         visualFramePosition=[0, 0, 0],
        #     )
        # p.createMultiBody(
        #     baseMass=0,  # purely visual
        #     baseVisualShapeIndex=visual_shape_id,
        #     basePosition=(self.goal_position[0], self.goal_position[1], 0.01),
        # )

        xy = [self.goal_position[0], self.goal_position[1]]  # Replace with your desired center
        z = 0.005     # Slightly above ground to avoid z-fighting

        radii = 1.5 * np.array([0.06, 0.045, 0.03, 0.015])
        heights = [0.01, 0.012, 0.014, 0.016]
        colors = [
            [0.0, 0.0, 0.0, 1.0],   # black (outer)
            [0.0, 0.0, 1.0, 1.0],   # blue
            [1.0, 0.0, 0.0, 1.0],   # red
            [1.0, 1.0, 0.0, 1.0],   # yellow (center)
        ]

        for r, h, c in zip(radii, heights, colors):
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=r,
                length=h,
                rgbaColor=c,
                visualFramePosition=[0, 0, 0],
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[xy[0], xy[1], z],
            )

        # color objects
        for entity_name, entity_id in self.sim_entities.items():
            print("entity_name: ", entity_name)
            if entity_id == self.cue_ball_id:
                p.changeVisualShape(objectUniqueId=entity_id, linkIndex=-1, rgbaColor=[0.7705299, 0.60735886, 0.1933879, 1])
            elif entity_id == self.target_ball_id:
                p.changeVisualShape(objectUniqueId=entity_id, linkIndex=-1, rgbaColor=[0.44015701, 0.53796967, 0.71892311, 1])
            elif entity_name == "ground_plane":
                p.changeVisualShape(objectUniqueId=entity_id, linkIndex=-1, textureUniqueId=-1, rgbaColor=[1., 1., 1., 1.])
            else:
                if entity_id != self.robot_id:
                    p.changeVisualShape(objectUniqueId=entity_id, linkIndex=-1, rgbaColor=[0.5, 0.5, 0.5, 1])


    def add_robot(self):
        # get robot information
        robot_position = self.config["scenario"]["robot"]["position"]
        joint_names = self.config["scenario"]["robot"]["joint_names"]
        default_joint_positions = self.config["scenario"]["robot"]["default_joint_positions"]

        # add robot to simulation
        robot_id = p.loadURDF(
            fileName=self.config["scenario"]["robot"]["urdf_path"],
            basePosition=robot_position,
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        # set friction and restitution for all links
        num_links = p.getNumJoints(robot_id)
        for link_index in range(-1, num_links):  # -1 includes the base link
            p.changeDynamics(
                bodyUniqueId=robot_id,
                linkIndex=link_index,
                lateralFriction=0.65,
                restitution=0.75,
            )

        # get list of joint names
        joint_name_to_id = {}
        num_joints = p.getNumJoints(robot_id)
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(robot_id, joint_index)
            joint_name = joint_info[1].decode("utf-8")
            joint_name_to_id[joint_name] = joint_index

        # robot-specific information
        self.robot_info = {
            "joint_names": joint_names,
            "dofs_idx":
                [joint_name_to_id[name] for name in joint_names],
            "default_joint_positions": default_joint_positions,
        }

        # "Unlock" the default controller
        for i in range(7):
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, force=0)

        # set initial joint positions
        for idx in self.robot_info["dofs_idx"]:
            p.resetJointState(robot_id, idx, default_joint_positions[idx])

        self.robot_id = robot_id

    def reset(self):
        if self.sim_reset_state is None:
            self.sim_reset_state = p.saveState()
        else:
            p.restoreState(self.sim_reset_state)
        
        if self.controllers is not None:
            for controller in self.controllers:
                controller.reset()

        # reset target ball trajectory
        self.target_ball_trajectory = []
        self.robot_joint_trajectory = []

    def add_controllers(self):
        n_envs = self.config["physics_simulation"]["n_envs"]
        self.controllers = [
            BilliardsManipulatorController(
                robot_urdf_path = self.config["scenario"]["robot"]["urdf_path"],
                joint_names = self.robot_info["joint_names"],
                control_dt = self.config["physics_simulation"]["sim_options"]["dt"],
                kp_pos = 100.0, # 50.0,
                kp_rot = 7.5, # 2.0,
                kd_pos = 45.0, # 20.0,
                kd_rot = 5.0, #1.0,
                kp_joint = 3.0, #1.0,
                kd_joint = 4.5, #0.5,
            ) for _ in range(n_envs)
        ]

    def apply_control(self):
        # TODO: parallelize this

        # get current state
        joint_states = p.getJointStates(self.robot_id, range(7))
        q = np.array([state[0] for state in joint_states])
        dq = np.array([state[1] for state in joint_states])

        # get controller commands
        for i, controller in enumerate(self.controllers):
            if controller.times is not None: # TODO: fix this
                tau = controller.get_command(q, dq)
                # velocities = controller.get_command(q, dq)
            else:
                tau = np.zeros_like(q[i])
                # velocities = np.zeros_like(q[i])

            # apply control

            p.setJointMotorControlArray(
                self.robot_id, list(range(7)), p.TORQUE_CONTROL, forces=tau
            )
            # p.setJointMotorControlArray(
            #     self.robot_id, list(range(7)), p.VELOCITY_CONTROL, targetVelocities=velocities
            # )
        
    def apply_configurations(self, configurations: np.ndarray):
        """
        Apply configurations to the cue ball.
        
        Args:
            configurations (np.ndarray): Array of shape (n, 2) where n is the number of configurations.
                Each configuration is a tuple of (launch velocity, launch angle).
        """
        # compute velocities from configuration as (n x (vx, vy, vz)) array
        v = configurations[:, 0]
        theta = configurations[:, 1]

        cue_ball_position = np.array(p.getBasePositionAndOrientation(self.cue_ball_id)[0])

        if self.use_robot:
            for i, (v, theta) in enumerate(zip(v, theta)):
                reference_motion = self._generate_strike_reference_motion(
                    contact_position=cue_ball_position,
                    contact_speed=v,
                    contact_angle=theta
                )
                self.controllers[i].set_reference_motion(reference_motion)
        else:
            vx = v * np.cos(theta)
            vy = v * np.sin(theta)
            vz = np.zeros_like(v)

            dof_velocities = np.column_stack((vx, vy, vz, np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)))

            # determine number of envs needed
            envs_idx = None
            if dof_velocities.shape[0] < self.scene.n_envs:
                envs_idx = list(range(dof_velocities.shape[0]))

            # apply respective velocities to each env's cue ball
            self.cue_ball.set_dofs_velocity(dof_velocities, envs_idx=envs_idx)

    def _generate_strike_reference_motion(
        self,
        contact_position: np.ndarray,
        contact_speed: float, 
        contact_angle: float,
        l_accel: float = 0.1,
        l_decel: float = 0.1,
    ):
        # print("contact_position: ", contact_position)
        # print("contact_speed: ", contact_speed)
        # print("contact_angle: ", np.degrees(contact_angle))

        # account for robot position offset
        robot_base_position = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        contact_position = contact_position - robot_base_position

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

        times = np.arange(curve.a, curve.b + self.dt, self.dt)
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
    
    def log_simulation_data(self):
        # log target ball trajectory
        target_ball_pos = p.getBasePositionAndOrientation(self.target_ball_id)[0]
        self.target_ball_trajectory.append(target_ball_pos)

        # log robot joint trajectory
        num_joints = p.getNumJoints(self.robot_id)
        joint_states = p.getJointStates(self.robot_id, list(range(num_joints)))
        joint_positions = [state[0] for state in joint_states]
        self.robot_joint_trajectory.append(joint_positions)
        

    def compute_task_loss(self):
        # compute distance between target ball and goal position
        target_ball_trajectory = np.array(self.target_ball_trajectory)

        # compute loss as best distance to goal position
        loss = np.linalg.norm(target_ball_trajectory[:, :2] - self.goal_position[:2], axis=1).min()

        return np.array([loss])

    def generate_configuration_grid(self):
        # compute vector from cue ball to target ball
        cue_ball_pos = np.array(p.getBasePositionAndOrientation(self.cue_ball_id)[0])
        target_ball_pos = np.array(p.getBasePositionAndOrientation(self.target_ball_id)[0])
        aim_vector = target_ball_pos - cue_ball_pos
        aim_vector = aim_vector / np.linalg.norm(aim_vector)

        # center the grid around this initial guess
        v_center = self.config["scenario"]["settings"]["launch_velocity"]
        theta_center = np.arctan2(aim_vector[1], aim_vector[0])

        # generate grid around center
        v_range = 0.6
        theta_range = 7 * np.pi / 180.0
        n = 61

        # v_sweep, theta_sweep = np.meshgrid(
        #     np.linspace(v_center - v_range, v_center + v_range, n),
        #     np.linspace(theta_center - theta_range, theta_center + theta_range, n),
        # )
        v_sweep, theta_sweep = np.meshgrid(
            np.linspace(0.25, 0.8, n),
            np.linspace(theta_center - theta_range, theta_center + theta_range, n),
        )
        v_sweep = v_sweep.flatten()
        theta_sweep = theta_sweep.flatten()
        configurations = np.stack([v_sweep, theta_sweep], axis=1)

        return configurations
    
    def optimize_plan(self):
        for iter in range(30):
            print("Running Initialization Iteration: ", iter)
            configurations = self.generate_configuration_grid()

            # sample n_envs from configurations randomly
            n_envs = self.config["physics_simulation"]["n_envs"]
            indices = np.random.choice(len(configurations), n_envs, replace=False)
            x_starts = configurations[indices]

            indices = np.random.choice(len(configurations), 3, replace=False)
            initial_simplex = np.array([configurations[indices]])

            opt_solution, opt_loss, trajectories, f_trajectories = nelder_mead(
                self.evaluate_rollouts,
                x_starts=x_starts,
                initial_simplexes=initial_simplex,
                scale_factors=np.array([0.1, 0.1]),
                bounds = np.array([[0.2, 0.85], [-np.pi/2, np.pi/2]]),
            )

            # create directory
            run_name = self.config["dataset_config"]["dataset_path"].split("/")[-1] 
            planning_directory = Path(self.config["save_directory"]) / run_name / "planning_results"
            # create_directory(planning_directory, overwrite=True)
            
            overwrite = True if iter == 0 else False
            create_directory(planning_directory, overwrite=overwrite)

            # package optimal action
            contact_position = np.array(p.getBasePositionAndOrientation(self.cue_ball_id)[0])
            action_dict = {
                "contact_position": contact_position.tolist(),
                "contact_speed": opt_solution[0],
                "contact_angle": opt_solution[1],
            }
            
            # save results
            opt_trajectory_file_path = get_unique_file_path(str(planning_directory) + f"/optimization_trajectory_{iter}.npy")
            opt_obj_vals_file_path = get_unique_file_path(str(planning_directory) + f"/optimization_objective_values_{iter}.npy")
            opt_solution_file_path = get_unique_file_path(str(planning_directory) + f"/opt_solution_{iter}.npy")
            opt_loss_file_path = get_unique_file_path(str(planning_directory) + f"/opt_loss_{iter}.npy")
            action_specification_file_path = get_unique_file_path(str(planning_directory) + f"/action_specification_{iter}.json")

            np.save(opt_trajectory_file_path, trajectories)
            np.save(opt_obj_vals_file_path, f_trajectories)
            np.save(opt_solution_file_path, opt_solution)
            np.save(opt_loss_file_path, opt_loss)
            save_dict_to_json(action_dict, Path(action_specification_file_path).name, directory=planning_directory)


        # np.save(str(planning_directory) + "/optimization_trajectory.npy", trajectories)
        # np.save(str(planning_directory) + "/optimization_objective_values.npy", f_trajectories)
        # np.save(str(planning_directory) + "/opt_solution.npy", opt_solution)
        # np.save(str(planning_directory) + "/opt_loss.npy", opt_loss)
        # save_dict_to_json(action_dict, "action_specification", directory=planning_directory)

        # print("Saving robot trajectory...")
        # self.evaluate_rollouts(opt_solution, reset=False)
        # np.save(str(planning_directory) + "/joint_trajectory.npy", np.array(self.robot_joint_trajectory))
        # breakpoint()


    
    
        




    


