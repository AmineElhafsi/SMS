from typing import Dict, List, Optional, Tuple

import omni
import omni.isaac.core.utils.prims as prims_utils
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import GeometryPrim, RigidPrim, XFormPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.types import ArticulationAction
from pxr import UsdPhysics

from materials.utils import apply_physics_material_to_prim
from scenes.base import BaseScene
from scenes.entities import SceneProp
from scenes.props_lists import BILLIARDS_OBSTACLES
from scenes.simple_room_helpers import randomly_place_billiard_balls, randomly_place_obstacles
from scenes.utils import debug_marker
from src.controllers.manipulator import BilliardsManipulatorController
from src.controllers.utils import generate_strike_reference_motion
from src.utils.io import create_directory

from robots.manipulators import FrankaFR3

class SimpleRoom(BaseScene):
    def __init__(self, config):
        super().__init__(config)

    def setup_base_scene(self):
        print("Setting up empty scene...")
        # load scene USD
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise FileNotFoundError("Could not find assets root path")
        usd_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"
        simple_room_prim = prims_utils.create_prim(
            prim_path="/World/SimpleRoom",
            prim_type="Xform",
            usd_path=usd_path,
        )
        self.world.render()

        # adjust scene height so that floor is at z=0
        simple_room_xform_prim = XFormPrim(
            prim_path="/World/SimpleRoom",
        )
        simple_room_xform_prim.set_world_pose(position=np.array([0., 0., 0.7695]))
        self.world.render()

        # set physics material for scene (wood for floor material; density not set as room is static)
        apply_physics_material_to_prim(simple_room_prim, "wood", set_density=False)
        self.world.render()


        # deactivate default table
        table_prim = prims_utils.get_prim_at_path("/World/SimpleRoom/table_low_327")
        table_prim.SetActive(False)
        self.world.render()

    def add_robot(self, robot):
        self.robot = robot
  
    def add_controller(self):
        # first, set stiffness and damping to 0 for all joints
        # to enable effort (torque) control
        stage = omni.usd.get_context().get_stage()
        for prim in stage.TraverseAll():
            prim_type = prim.GetTypeName()
            if prim_type in ["PhysicsRevoluteJoint" , "PhysicsPrismaticJoint"]:
                if prim_type == "PhysicsRevoluteJoint":
                    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                else:
                    drive = UsdPhysics.DriveAPI.Get(prim, "linear")
                drive.GetStiffnessAttr().Set(0.0)
                drive.GetDampingAttr().Set(0.0)

        # get robot config
        robot_config = self.config["isaac"]["robot"]

        # create controller
        self.controller = BilliardsManipulatorController(
            robot_urdf_path = robot_config["urdf_path"],
            joint_names = robot_config["joint_names"],
            control_dt = self.dt,
            kp_pos = 3*1100.0,
            kp_rot = 5*350.0,
            kd_pos = 850.0,
            kd_rot = 550.0, # 5.0,
            kp_joint = [250.0, 250.0, 250.0, 250.0, 250.0, 1000.0, 1000.0],
            kd_joint = [550.0, 550.0, 550.0, 550.0, 550.0, 2000.0, 2000.0], # 4.5,
        )

    def simulate_strike(
        self,
        contact_position: np.ndarray,
        contact_speed: float,
        contact_angle: float,
        joint_trajectory: List[np.ndarray] = None,
        record: bool = False,
    ):
        
        if record:
            import cv2
            import pickle
            from pathlib import Path
            from sensors.cameras import OrbbecGemini2
            # create camera
            camera = OrbbecGemini2(self.world)

            # set image framing
            # camera.set_world_pose(position=np.array([1.46998, 1.65188, 1.56756]))
            camera.set_world_pose(position=np.array([1.21427, 1.11677, 1.1393]))
            camera.point_at(np.array([0.5, 0.0, 0.0]))
            self.world.render()

            frame_directory = Path("manipulator_frames")
            create_directory(frame_directory, overwrite=False)

        # move contact position to be relative to robot base
        contact_position = contact_position - np.array([-0.1, 0.0, 0.0])

        # provide reference motion for controller to track
        reference_motion = generate_strike_reference_motion(
            contact_position=contact_position,
            contact_speed=contact_speed,
            contact_angle=contact_angle,
            dt=self.dt,
        )
        self.controller.reset()
        self.controller.set_reference_motion(reference_motion)

        # get controlled joint indices
        robot_config = self.config["isaac"]["robot"]
        joint_names = robot_config["joint_names"]
        controlled_joint_indices = [self.robot.robot.dof_names.index(joint_name) for joint_name in joint_names]

        # store target ball path
        target_ball_positions = [self.scene_props["target_ball"].get_world_position()]

        # run simulation
        prev_t = self.world.current_time
        sim_duration = 8.0 #self.config["isaac"]["sim_duration"]
        num_time_steps = int(sim_duration / self.dt)
        for step in range(num_time_steps):
            
            if record and step % 50 == 0:
                current_frame = camera.get_data(render_steps=30)
                color = current_frame["rgba"][:, :, :3]
                instance_segmentation_dict = current_frame["instance_segmentation"]
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_directory / f"{step:04d}.png"), color)
                # save pkl of instance_segmentation_dict
                with open(str(frame_directory / f"{step:04d}_seg.pkl"), 'wb') as f:
                    pickle.dump(instance_segmentation_dict, f)
                
            # get current state
            q = self.robot.robot.get_joint_positions()[controlled_joint_indices]
            dq = self.robot.robot.get_joint_velocities()[controlled_joint_indices]

            # get control command
            tau = self.controller.get_command(q, dq)
            # velocities = self.controller.get_command(q, dq)

            # apply control command
            # action = ArticulationAction(
            #     joint_velocities = velocities, 
            #     joint_indices = controlled_joint_indices
            # )
            # self.robot.robot.apply_action(action)
            self.robot.robot.set_joint_efforts(tau, joint_indices=controlled_joint_indices)
            # self.robot.robot.set_joint_velocities(velocities, joint_indices=controlled_joint_indices)
            
            # self.robot.robot.set_joint_velocity_targets(velocities, joint_indices=controlled_joint_indices)
            self.robot.robot.set_joint_positions(np.zeros(2), joint_indices=[7, 8]) # keeps gripper closed
            
            self.step()

            target_ball_positions.append(self.scene_props["target_ball"].get_world_position())

            # print("Current Time: ", self.world.current_time)
            prev_t = self.world.current_time

        return target_ball_positions

    def generate_billiards_scenario(self, visualize: bool = False):
        # parameters
        r_cue = 0.3
        r_target_min = 0.45
        r_target_max = 0.7
        r_camera = 1.0
        r_goal_arc = 1.1

        # r_obstacle_min = 0.45
        # r_obstacle_max = 0.65
        # max_r_goal = 1.25
        ball_scale_factor = 1.3
        ball_radius = 0.03 * ball_scale_factor

        dtheta_cue = np.radians(20)#30)
        dtheta_obstacle_ball = np.radians(20) #40)
        dtheta_obstacle_object = np.radians(90)
        dtheta_goal = np.radians(75)

        theta_buffer = np.radians(5)

        # set cue ball position
        theta_cue = np.random.uniform(-dtheta_cue, dtheta_cue)
        x_cue = r_cue * np.cos(theta_cue)
        y_cue = r_cue * np.sin(theta_cue)
        cue_ball_position = np.array([x_cue, y_cue, ball_radius])
        cue_ball = SceneProp(
            usd_path="file:/home/anonymous/Documents/Research/simulation/assets/props/billiard_balls/BilliardBalls_CueBall.usd",
            group="BilliardBalls",
            position=cue_ball_position,
            scale_factor=ball_scale_factor,
            enable_physics=True,
            physics_material_name="resin"
        )
        self.scene_props["cue_ball"] = cue_ball

        # set target ball position
        r_target = np.random.uniform(r_target_min, r_target_max)
        theta_target = np.random.uniform(-dtheta_obstacle_ball, dtheta_obstacle_ball)
        x_target = r_target * np.cos(theta_target)
        y_target = r_target * np.sin(theta_target)
        target_ball_position = np.array([x_target, y_target, ball_radius])
        target_ball_number = np.random.choice(range(1, 16))
        target_ball = SceneProp(
            usd_path=f"file:/home/anonymous/Documents/Research/simulation/assets/props/billiard_balls/BilliardBalls_{target_ball_number:02}.usd",
            group="BilliardBalls",
            position=target_ball_position,
            scale_factor=ball_scale_factor,
            enable_physics=True,
            physics_material_name="resin"
        )
        self.scene_props["target_ball"] = target_ball

        # set goal position
        incident_vector = target_ball_position - cue_ball_position
        incident_distance = np.linalg.norm(incident_vector)
        incident_angle = np.arctan2(incident_vector[1], incident_vector[0])

        outgoing_angle = np.random.uniform(incident_angle - dtheta_goal, incident_angle + dtheta_goal)

        # compute distance between target ball and goal (on circle of radius r_goal)
        # assuming target ball moves along outgoing_angle
        
        a = 1
        b = 2 * np.cos(outgoing_angle) * (target_ball_position[0]) + np.sin(outgoing_angle) * (target_ball_position[1])
        c = target_ball_position[0]**2 + target_ball_position[1]**2 - r_goal_arc**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            raise ValueError("Discriminant is negative")
        t1 = (-b + np.sqrt(discriminant)) / (2*a)
        t2 = (-b - np.sqrt(discriminant)) / (2*a)
        r_goal = max(t1, t2)

        # r_goal = np.random.uniform(1.1, 1.25)
        # r_goal = np.random.uniform(0.2, 0.65)
        print("r_goal: ", r_goal)
        goal_position = r_goal * np.array([np.cos(outgoing_angle), np.sin(outgoing_angle), ball_radius]) + target_ball_position
        outgoing_vector = goal_position - target_ball_position
        outgoing_distance = np.linalg.norm(outgoing_vector)
        print("Outgoing distance: ", outgoing_distance)

        ########### draw incident and outgoing vectors ############
        if visualize:
            r_buffer = np.linalg.norm(target_ball_position - cue_ball_position)
            cue_ball_left_buffer = np.array(
                [r_buffer * np.cos(incident_angle - theta_buffer), r_buffer * np.sin(incident_angle - theta_buffer), 0.03]
            )
            cue_ball_right_buffer = np.array(
                [r_buffer * np.cos(incident_angle + theta_buffer), r_buffer * np.sin(incident_angle + theta_buffer), 0.03]
            )
            cue_ball_left_buffer_wide = np.array(
                [r_buffer * np.cos(incident_angle - theta_buffer - dtheta_obstacle_ball), r_buffer * np.sin(incident_angle - theta_buffer - dtheta_obstacle_ball), 0.03]
            )
            cue_ball_right_buffer_wide = np.array(
                [r_buffer * np.cos(incident_angle + theta_buffer + dtheta_obstacle_ball), r_buffer * np.sin(incident_angle + theta_buffer + dtheta_obstacle_ball), 0.03]
            )

            incident_pts = np.linspace(cue_ball_position, target_ball_position, 100)
            incident_pts_left = np.linspace(cue_ball_position, cue_ball_position+cue_ball_left_buffer, 100)
            incident_pts_right = np.linspace(cue_ball_position, cue_ball_position+cue_ball_right_buffer, 100)
            incident_pts_left_wide = np.linspace(cue_ball_position, cue_ball_position+cue_ball_left_buffer_wide, 100)
            incident_pts_right_wide = np.linspace(cue_ball_position, cue_ball_position+cue_ball_right_buffer_wide, 100)
            
            for pt in incident_pts:
                debug_marker(pt, color=np.array([255., 255., 0.]), size=0.01)
            for pt in incident_pts_left:
                debug_marker(pt, color=np.array([.75, .1, .8]), size=0.01)
            for pt in incident_pts_right:
                debug_marker(pt, color=np.array([.75, .1, .8]), size=0.01)
            for pt in incident_pts_left_wide:
                debug_marker(pt, color=np.array([.75, .1, .8]), size=0.01)
            for pt in incident_pts_right_wide:
                debug_marker(pt, color=np.array([.75, .1, .8]), size=0.01)

            r_buffer = np.linalg.norm(goal_position - target_ball_position)
            target_ball_left_buffer = np.array(
                [r_buffer * np.cos(outgoing_angle - theta_buffer), r_buffer * np.sin(outgoing_angle - theta_buffer), 0.03]
            )
            target_ball_right_buffer = np.array(
                [r_buffer * np.cos(outgoing_angle + theta_buffer), r_buffer * np.sin(outgoing_angle + theta_buffer), 0.03]
            )
            target_ball_left_buffer_wide = np.array(
                [r_buffer * np.cos(outgoing_angle - theta_buffer - dtheta_obstacle_object), r_buffer * np.sin(outgoing_angle - theta_buffer - dtheta_obstacle_object), 0.03]
            )
            target_ball_right_buffer_wide = np.array(
                [r_buffer * np.cos(outgoing_angle + theta_buffer + dtheta_obstacle_object), r_buffer * np.sin(outgoing_angle + theta_buffer + dtheta_obstacle_object), 0.03]
            )

            outgoing_pts = np.linspace(target_ball_position, goal_position, 100)
            outgoing_pts_left = np.linspace(target_ball_position, target_ball_position+target_ball_left_buffer, 100)
            outgoing_pts_right = np.linspace(target_ball_position, target_ball_position+target_ball_right_buffer, 100)
            outgoing_pts_left_wide = np.linspace(target_ball_position, target_ball_position+target_ball_left_buffer_wide, 100)
            outgoing_pts_right_wide = np.linspace(target_ball_position, target_ball_position+target_ball_right_buffer_wide, 100)

            for pt in outgoing_pts:
                debug_marker(pt, color=np.array([135., 206., 235.]), size=0.01)
            for pt in outgoing_pts_left:
                debug_marker(pt, color=np.array([.75, .5, .75]), size=0.01)
            for pt in outgoing_pts_right:
                debug_marker(pt, color=np.array([.75, .5, .75]), size=0.01)
            for pt in outgoing_pts_left_wide:
                debug_marker(pt, color=np.array([.75, .5, .75]), size=0.01)
            for pt in outgoing_pts_right_wide:
                debug_marker(pt, color=np.array([.75, .5, .75]), size=0.01)

        ############################################################
        # place obstacles
        # obstacles must avoid the area defined by the cue ball, target ball, and goal
        # additionally, the incident and outgoing vectors must not intersect with any obstacles

        # start with distractor billiard balls around incident path (these are allowed to intersect with incident path)
        randomly_place_billiard_balls(
            ball_numbers=list(set(range(1, 16)) - {target_ball_number}),
            max_balls=2,
            r_range=(0, incident_distance),
            theta_range=(incident_angle-dtheta_obstacle_ball, incident_angle+dtheta_obstacle_ball),
            ball_buffer_margin=2*ball_radius,
            offset_position=cue_ball_position,
            world=self.world,
            existing_scene_props=self.scene_props,
        )
        print("self.scene_props (billiards 1): ", self.scene_props.keys())

        # next place distractor billiard balls around outgoing path (also allowed to intersect with outgoing path)
        randomly_place_billiard_balls(
            ball_numbers=list(set(range(1, 16)) - {target_ball_number}),
            max_balls=2,
            r_range=(0, outgoing_distance),
            theta_range=(outgoing_angle-dtheta_obstacle_ball, outgoing_angle+dtheta_obstacle_ball),
            ball_buffer_margin=2*ball_radius,
            offset_position=target_ball_position,
            world=self.world,
            existing_scene_props=self.scene_props,
        )
        print("self.scene_props (billiards 2): ", self.scene_props.keys())

        # then place larger obstacles around the incident path (these are NOT allowed to intersect with outgoing path)
        randomly_place_obstacles(
            obstacle_choices=BILLIARDS_OBSTACLES,
            max_obstacles=4,
            r_range=(0, incident_distance),
            theta_range=(incident_angle-dtheta_obstacle_object, incident_angle+dtheta_obstacle_object),
            prop_buffer_margin=0.05,
            offset_position=target_ball_position,
            world=self.world,
            existing_scene_props=self.scene_props,
        )
        print("self.scene_props (obstacles 2): ", self.scene_props.keys())

        # finally place larger obstacles around the outgoing path (these are NOT allowed to intersect with outgoing path)
        randomly_place_obstacles(
            obstacle_choices=BILLIARDS_OBSTACLES,
            max_obstacles=4,
            r_range=(0, r_camera-0.1),#outgoing_distance),
            theta_range=(outgoing_angle-dtheta_obstacle_object, outgoing_angle+dtheta_obstacle_object),
            prop_buffer_margin=0.05,
            offset_position=target_ball_position,
            world=self.world,
            existing_scene_props=self.scene_props,
        )
        print("self.scene_props (obstacles 2): ", self.scene_props.keys())

        return cue_ball_position, target_ball_position, goal_position
