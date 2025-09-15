from pathlib import Path
from typing import Dict, List, Optional, Union

import genesis as gs
import numpy as np
import yaml

from src.controllers.manipulator import BilliardsManipulatorController
from src.physics.environments.base import BaseEnv
from src.physics.sim_entities import load_sim_object, Sphere, Mesh
from src.opspace.trajectories.bezier import bezier_trajectory
from src.opspace.trajectories.splines import CompositeBezierCurve
from src.optimization.nelder_mead import nelder_mead
from src.utils.io import create_directory, save_dict_to_json


class Billiards(BaseEnv):
    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__("billiards", config)

        # configure robot
        self.use_robot = config["scenario"]["settings"]["use_robot"]
        if self.use_robot:
            self.add_robot()
            self.add_controllers()

        # build scene
        self.build()

        # parse task
        self.parse_task()

        # reset / initialize environment    
        self.reset()

        breakpoint()
            
    def parse_task(self) -> Dict:
        # load task config
        scenario_config_path = Path(self.config["dataset_config"]["dataset_path"]) / "scenario_config.yaml"
        with open(scenario_config_path, "r") as file:
            scenario_config = yaml.safe_load(file)
        task_config = scenario_config["task"]

        # assemble list of balls among scene entities
        billiard_balls = []
        for entity_name, entity in self.sim_entities.items():
            if type(entity.morph) == gs.options.morphs.Sphere:
                ball_pos = entity.get_pos().cpu().numpy()[0]
                billiard_balls.append((entity_name, ball_pos))

        # identify cue ball
        cue_ball_pos = np.array(task_config["cue_ball_position"])
        cue_ball_name, cue_ball_pos = min(billiard_balls, key=lambda x: np.linalg.norm(x[1] - cue_ball_pos))

        # identify target ball
        target_ball_pos = np.array(task_config["target_ball_position"])
        target_ball_name, target_ball_pos = min(billiard_balls, key=lambda x: np.linalg.norm(x[1] - target_ball_pos))

        # define task-relevant variables
        self.cue_ball = self.sim_entities[cue_ball_name]
        self.target_ball = self.sim_entities[target_ball_name]
        self.goal_position = np.array(task_config["goal_position"])

        # add visual for goal
        goal_vis_sphere = self.scene.draw_debug_sphere(
            pos=self.goal_position, radius=0.025, color=(1, 0, 0, 0.5))  # Red with alpha

    def add_robot(self):
        # add robot to scene
        robot_position = self.config["scenario"]["robot"]["position"]
        joint_names = self.config["scenario"]["robot"]["joint_names"]
        default_joint_positions = self.config["scenario"]["robot"]["default_joint_positions"]

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file = self.config["scenario"]["robot"]["urdf_path"],
                pos=robot_position,
                fixed=True,
                links_to_keep=joint_names,
            ),
            material=gs.materials.Rigid(
                friction=0.65, 
                coup_restitution=0.75,
            )
        )

        # robot-specific information
        self.robot_info = {
            "joint_names": joint_names,
            "dofs_idx":
                [self.robot.get_joint(name).dof_idx_local for name in joint_names],
            "default_joint_positions": default_joint_positions,
        }

    def reset(self):
        # reset scene
        self.scene.reset()

        if self.use_robot:
            # reset robot to default configuration
            self.robot.set_dofs_position(
                self.robot_info["default_joint_positions"], 
                self.robot_info["dofs_idx"]
            )

            # reset controller
            for controller in self.controllers:
                controller.reset()

    def add_controllers(self):
        n_envs = self.config["physics_simulation"]["n_envs"]
        self.controllers = [
            BilliardsManipulatorController(
                robot_urdf_path = self.config["scenario"]["robot"]["urdf_path"],
                joint_names = self.robot_info["joint_names"],
                control_dt = self.config["physics_simulation"]["sim_options"]["dt"],
            ) for _ in range(n_envs)
        ]

    def apply_control(self):
        # TODO: parallelize this

        # get current state
        q = self.robot.get_dofs_position().cpu().numpy()[:, self.robot_info["dofs_idx"]]
        dq = self.robot.get_dofs_velocity().cpu().numpy()[:, self.robot_info["dofs_idx"]]

        # get controller commands
        tau = []
        for i, controller in enumerate(self.controllers):
            if controller.times is not None: # TODO: fix this
                tau.append(controller.get_command(q[i], dq[i]))
            else:
                tau.append(np.zeros_like(q[i]))
        tau = np.array(tau)

        # apply torques to robot
        self.robot.control_dofs_force(
            tau,
            self.robot_info["dofs_idx"],
        )

        # self.robot.control_dofs_velocity(
        #     tau,
        #     self.robot_info["dofs_idx"],
        # )

        # keep grippers closed
        gripper_target_positions = np.zeros((self.scene.n_envs, 2))
        self.robot.control_dofs_position(
            gripper_target_positions,
            #np.zeros(2),
            [7, 8],
        )

        
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

        if self.use_robot:
            for i, (v, theta) in enumerate(zip(v, theta)):
                reference_motion = self._generate_strike_reference_motion(
                    contact_position=self.cue_ball.get_pos().cpu().numpy()[0],#np.array([0.45, 0.0, self.cue_ball.get_pos().cpu().numpy()[0, -1]]),
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
        contact_position = contact_position - self.robot.get_pos().cpu().numpy()[0]

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

        times = np.arange(curve.a, curve.b + self.scene.dt, self.scene.dt)
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

    def compute_task_loss(self):
        return np.linalg.norm(self.target_ball.get_pos().cpu().numpy() - self.goal_position, axis=1)

    def generate_configuration_grid(self):
        # compute vector from cue ball to target ball
        cue_ball_pos = self.cue_ball.get_pos().cpu().numpy()[0]
        target_ball_pos = self.target_ball.get_pos().cpu().numpy()[0]
        aim_vector = target_ball_pos - cue_ball_pos
        aim_vector = aim_vector / np.linalg.norm(aim_vector)

        # center the grid around this initial guess
        v_center = self.config["scenario"]["settings"]["launch_velocity"]
        theta_center = np.arctan2(aim_vector[1], aim_vector[0])

        # generate grid around center
        v_range = 0.6
        theta_range = 7 * np.pi / 180.0
        n = 61

        v_sweep, theta_sweep = np.meshgrid(
            np.linspace(v_center - v_range, v_center + v_range, n),
            np.linspace(theta_center - theta_range, theta_center + theta_range, n),
        )
        v_sweep = v_sweep.flatten()
        theta_sweep = theta_sweep.flatten()
        configurations = np.stack([v_sweep, theta_sweep], axis=1)

        return configurations

    def optimize_plan(self):
        configurations = self.generate_configuration_grid()

        # sample n_envs from configurations randomly
        indices = np.random.choice(len(configurations), self.scene.n_envs, replace=False)
        x_starts = configurations[indices]

        opt_solution, opt_loss, trajectories, f_trajectories = nelder_mead(
            self.evaluate_rollouts,
            x_starts=x_starts,
            scale_factors=np.array([0.1, 0.1]),
            bounds = np.array([[0.2, 1.0], [-np.pi/2, np.pi/2]]),
        )

        # create directory
        run_name = self.config["dataset_config"]["dataset_path"].split("/")[-1] 
        planning_directory = Path(self.config["save_directory"]) / run_name / "planning"
        create_directory(planning_directory, overwrite=True)

        # package optimal action
        action_dict = {
            "contact_position": self.cue_ball.get_pos().cpu().numpy()[0].tolist(),
            "contact_speed": opt_solution[0],
            "contact_angle": opt_solution[1],
        }
        
        # save results
        np.save(str(planning_directory) + "/optimization_trajectory.npy", trajectories)
        np.save(str(planning_directory) + "/optimization_objective_values.npy", f_trajectories)
        np.save(str(planning_directory) + "/opt_solution.npy", opt_solution)
        np.save(str(planning_directory) + "/opt_loss.npy", opt_loss)
        save_dict_to_json(action_dict, "action_specification", directory=planning_directory)


    
    
        




    


