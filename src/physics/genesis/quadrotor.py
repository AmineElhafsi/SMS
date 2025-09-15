from pathlib import Path
from typing import Dict, List, Optional, Union

import genesis as gs
import numpy as np
import yaml

from src.controllers.manipulator import BilliardsManipulatorController
from src.physics.genesis.base import BaseEnv
from src.physics.sim_entities import load_sim_object, Sphere, Mesh
from src.opspace.trajectories.bezier import BezierCurve
from src.opspace.trajectories.splines import CompositeBezierCurve
from src.optimization.nelder_mead import nelder_mead
from src.utils.io import create_directory, save_dict_to_json
from mpl_toolkits.mplot3d import Axes3D


class QuadrotorLanding(BaseEnv):
    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__("quadrotor_landing", config)

        # configure robot
        self.use_robot = config["scenario"]["settings"]["use_robot"]
        if self.use_robot:
            self.add_robot()
            self.add_controllers()

        # build scene
        self.build()

        # parse task
        self.parse_task()

        # set phase (there are two phases: (1) landing, (2) approach)
        self.phase = "approach"

        # reset / initialize environment    
        self.reset()        
            
    def parse_task(self) -> Dict:
        # self.landing_target = self.sim_entities["a cardboard box_1"]
        landing_target_name = self.config["scenario"]["landing_target"]
        self.landing_target = self.sim_entities[landing_target_name] #["a cardboard box_1"]
        self.landing_height = 0.35

        # log data for loss computation
        self.entity_init_pos = {}
        self.entity_init_quat = {}
        landing_target_aabb = self.landing_target.get_AABB().cpu().numpy()
        z_min = landing_target_aabb[0][2]
        z_max = landing_target_aabb[1][2]
        z_mid = (z_min + z_max) / 2
        self.landing_height_threshold = z_mid
        self.flight_altitude = z_max + self.landing_height
        for k, v in self.sim_entities.items():
            self.entity_init_pos[k] = v.get_pos().cpu().numpy()
            self.entity_init_quat[k] = v.get_quat().cpu().numpy()

        # draw visuals of aabb extents
        # get corners of aabb
        x_min = landing_target_aabb[0][0]
        x_max = landing_target_aabb[1][0]
        y_min = landing_target_aabb[0][1]
        y_max = landing_target_aabb[1][1]
        z_min = landing_target_aabb[0][2]
        z_max = landing_target_aabb[1][2]

        # get corners of aabb
        corners = np.array([
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max],
        ])

        # for corner in corners:
        #     self.scene.draw_debug_sphere(pos=corner, radius=0.025, color=(1, 0, 0, 0.5))

    def add_robot(self):
        # robot-specific information
        self.robot_info = self.config["scenario"]["robot"]

        # create robot and set properties to match real quad
        scale = self.robot_info["size_x"] / 0.12 # match x-y footprint
        self.robot = self.scene.add_entity(
            morph=gs.morphs.Drone(
                scale=scale,
                file="urdf/drones/cf2x.urdf",
                pos=self.robot_info["initial_position"],
            ),
            material=gs.materials.Rigid(
                friction=0.6,
            )
        )

        # self.robot = self.scene.add_entity(
        #     morph=gs.morphs.Box(
        #         pos=self.robot_info["initial_position"],
        #         size=(self.robot_info["size_x"], self.robot_info["size_y"], self.robot_info["size_z"]),
        #     ),
        #     material=gs.materials.Rigid(
        #         friction=4.5,
        #     ),
        #     surface=gs.surfaces.Default(
        #         color    = (0.0, 0.0, 0.0),
        #         vis_mode = 'visual',
        #     ),
        # )

        # set up downwash model
        # determine emitter (propeller) parameters
        r_prop = 0.137*0.8 / 2 # m
        rho_air = 1.225 # kg/m^3
        m_quadrotor = 1.182 # kg
        n_propellers = 4 
        A_prop = np.pi * r_prop**2 # m^2

        F_prop = m_quadrotor * 9.81 / n_propellers # N
        v_0 = 0.0 # m/s
        v_exit = np.sqrt(F_prop / (0.5 * rho_air * A_prop))
        v_prop = 0.5 * (v_0 + v_exit) # m/s

        mass_flow_propeller = rho_air * A_prop * v_prop # kg/s

        n_particles = 6
        particle_length = 0.015
        particle_diameter = 0.025
        particle_volume = np.pi * (particle_diameter / 2)**2 * particle_length # m^3
        rho_particle = mass_flow_propeller / (n_particles * particle_volume) # kg/m^3

        print("Rho Particle: ", rho_particle)
        # rho_particle = 100.0 # kg/m^3
        self.emitters = [
            self.scene.add_emitter(
                material=gs.materials.SPH.Liquid(
                    rho=rho_particle, #150.0,
                    sampler="regular"
                ),
                max_particles=50,
                surface=gs.surfaces.Glass(
                    color=(0.7, 0.85, 1.0, 0.7),
                ),
            ) for _ in range(4)
        ]
        
    def reset(self):
        # reset scene
        self.scene.reset()

        physics_config = self.config["physics_simulation"]

        # reset robot to default configuration
        initial_approach_position = self.robot_info["initial_position"]
        initial_approach_position[2] = self.flight_altitude
        self.robot.set_dofs_position(
            initial_approach_position,
            dofs_idx_local=[0, 1, 2]
        )
        self.robot.links[0].set_mass(self.robot_info["mass"])
        for k, v in self.sim_entities.items():
            if k in self.entity_masses.keys():
                v.links[0].set_mass(self.entity_masses[k])
                print("Set mass of ", k, " to ", self.entity_masses[k])
            else:
                print("Mass of ", k, "is", self.sim_entities[k].links[0].get_mass())


            # print("Entity Name: ", k)
            # print("Object Mass: ", v.links[0].get_mass())
            # v.links[0].set_mass(4.5)

        # self.sim_entities["a cardboard box_1"].links[0].set_mass(1.0)
        # self.sim_entities["tower_base"].links[0].set_mass(4.43)
        # self.sim_entities["tower_roof"].links[0].set_mass(4.48)
        
    def add_controllers(self):
        pass

    def apply_control(self):
        t = self.scene.cur_t

        time_scaling = self.approach_path.arc_length() / self.robot_info["approach_velocity"]
        scaled_t = t / time_scaling

        # if we've reached our destination, stop moving, stop propellers, and land
        if scaled_t > 1.0:
            return

        # otherwise, move quadrotor to new position along approach path
        path_position = self.approach_path(scaled_t)
        updated_quadrotor_position = np.array([path_position[0], path_position[1], self.flight_altitude])
        quadrotor_orientation = np.zeros(3)
        updated_quadrotor_dofs = np.concatenate(
            (updated_quadrotor_position, quadrotor_orientation),
            axis=0
        )
        self.robot.set_dofs_position(
            updated_quadrotor_dofs,
            # dofs_idx_local=[0, 1, 2],
        )

        # place emitters at quadrotor corners
        size_x = self.robot_info["size_x"]
        size_y = self.robot_info["size_y"]
        size_z = self.robot_info["size_z"]

        eps_z = 0.02
        # corner_offsets = [
        #     np.array([size_x / 2, size_y / 2, -size_z / 2 - eps_z]),   # front-right corner
        #     np.array([size_x / 2, -size_y / 2, -size_z / 2 - eps_z]),  # front-left corner
        #     np.array([-size_x / 2, size_y / 2, -size_z / 2 - eps_z]),  # back-right corner
        #     np.array([-size_x / 2, -size_y / 2, -size_z / 2 - eps_z])  # back-left corner
        # ]
        corner_offsets = [
            np.array([size_x / 3, size_y / 3, -size_z / 2 - eps_z]),   # front-right corner
            np.array([size_x / 3, -size_y / 3, -size_z / 2 - eps_z]),  # front-left corner
            np.array([-size_x / 3, size_y / 3, -size_z / 2 - eps_z]),  # back-right corner
            np.array([-size_x / 3, -size_y / 3, -size_z / 2 - eps_z])  # back-left corner
        ]
        corner_positions = [updated_quadrotor_position + offset for offset in corner_offsets]

        # determin emitter (propeller) parameters
        r_prop = 0.137*0.8 / 2 # m
        rho_air = 1.225 # kg/m^3
        m_quadrotor = 1.182 # kg
        n_propellers = 4 
        A_prop = np.pi * r_prop**2 # m^2

        F_prop = m_quadrotor * 9.81 / n_propellers # N
        v_0 = 0.0 # m/s
        v_exit = np.sqrt(F_prop / (0.5 * rho_air * A_prop))
        v_prop = 0.5 * (v_0 + v_exit) # m/s

        mass_flow_propeller = rho_air * A_prop * v_prop # kg/s

        n_particles = 6
        particle_length = 0.015
        particle_diameter = 0.025
        particle_volume = np.pi * (particle_diameter / 2)**2 * particle_length # m^3
        rho_particle = mass_flow_propeller / (n_particles * particle_volume) # kg/m^3

        # print("v_prop: ", v_prop)
        # print("particle_")
        # v_prop = 1.5
        for i, emitter in enumerate(self.emitters):
            emitter.emit(
                pos=corner_positions[i],
                direction=np.array([0.0, 0, -1.0]),
                speed=v_prop,
                droplet_shape="circle",
                droplet_size=0.025, #particle_diameter,
            )

        # self.emitters[0].emit(
        #     pos=self.robot.get_pos().cpu().numpy()[:3],
        #     direction=np.array([0.0, 0, -1.0]),
        #     speed=1.5,
        #     droplet_shape="circle",
        #     droplet_size=0.05,
        #     # droplet_length=0.05
        # )

    def apply_configurations(self, configuration: np.ndarray):
        # unpack configurations
        approach_radius = configuration[0, 0]
        approach_angle = configuration[0, 1]
        landing_x = configuration[0, 2]
        landing_y = configuration[0, 3]
        
        # generate approach path
        quadrotor_position = np.array(self.robot_info["initial_position"][:2])
        landing_position = np.array([landing_x, landing_y])
        approach_control_point = landing_position + approach_radius * np.array([np.cos(approach_angle), np.sin(approach_angle)])
        self.approach_path = BezierCurve(
            np.array([quadrotor_position, approach_control_point, landing_position]),
            a=0.0,
            b=1.0,
        )

        self.landing_position_3D = np.concatenate(
            (landing_position, np.array([self.flight_altitude - self.landing_height])),
            axis=0
        )

    def compute_task_loss(self):
        total_loss = 0.0

        # compute loss for entity position movement
        position_deviation_loss = 0.0
        for k, v in self.sim_entities.items():
            if k == "ground_plane":
                continue
            entity_final_position = v.get_pos().cpu().numpy()
            entity_initial_position = self.entity_init_pos[k]
            position_deviation_loss += np.linalg.norm(entity_final_position - entity_initial_position)
        total_loss += position_deviation_loss

        print("Position Deviation Loss: ", position_deviation_loss)

        # compute loss for entity orientation movement
        orientation_deviation_loss = 0.0
        for k, v in self.sim_entities.items():
            if k == "ground_plane":
                continue
            entity_final_orientation = v.get_quat().cpu().numpy()
            entity_initial_orientation = self.entity_init_quat[k]

            dot_product = np.sum(entity_final_orientation * entity_initial_orientation)
            if np.any((dot_product < -1) | (dot_product > 1)):
                print("Warning: dot product out of bounds")
            orientation_deviation_loss += 2 * np.arccos(np.abs(dot_product))
        total_loss += orientation_deviation_loss

        print("Orientation Deviation Loss: ", orientation_deviation_loss)

        # compute loss for quadrotor landing accuracy
        landing_accuracy_loss = 0.0
        landing_accuracy_loss = np.linalg.norm(
            self.robot.get_pos().cpu().numpy() - self.landing_position_3D
        )
        total_loss += landing_accuracy_loss

        print("Landing Accuracy Loss: ", landing_accuracy_loss)

        # compute loss for landing site centering on target
        # landing_site_centering_loss = 0.0
        # landing_target_aabb = self.landing_target.get_AABB().cpu().numpy()
        # x_min = landing_target_aabb[0][0]
        # x_max = landing_target_aabb[1][0]
        # y_min = landing_target_aabb[0][1]
        # y_max = landing_target_aabb[1][1]
        # landing_site_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
        # landing_site_centering_loss = np.linalg.norm(landing_site_center - self.landing_position_3D[:2])
        # total_loss += 0.25 * landing_site_centering_loss
        
        # quadrotor_altitude_loss = 0.0
        # success_landing_mask = quadrotor_final_altitude > self.landing_height_threshold
        # quadrotor_altitude_loss[success_landing_mask] = 0.0
        # quadrotor_altitude_loss[~success_landing_mask] = np.inf
        # total_loss += quadrotor_altitude_loss
        print("Landing Success: ", landing_accuracy_loss < 0.2)
        
        return [total_loss]

    def sample_configuration(self):
        # assume landing target is flat, get x, y extents
        landing_target_aabb = self.landing_target.get_AABB().cpu().numpy()
        
        margin = 0.0 # 0.25

        x_min = landing_target_aabb[0][0] + margin
        x_max = landing_target_aabb[1][0] - margin
        y_min = landing_target_aabb[0][1] + margin
        y_max = landing_target_aabb[1][1] - margin

        # plot corners of aabb as debug visuals
        corners = np.array([
            [x_min, y_min, self.flight_altitude],
            [x_min, y_max, self.flight_altitude],
            [x_max, y_min, self.flight_altitude],
            [x_max, y_max, self.flight_altitude],
        ])

        # for corner in corners:
        #     self.scene.draw_debug_sphere(pos=corner, radius=0.025, color=(0, 0, 1, 0.5))



        # configuration is a 4-tuple (approach_radius, approach_angle, landing_x, landing_y)
        approach_radius = np.random.uniform(0.5, 2.5)
        approach_angle = np.random.uniform(0, 2 * np.pi)
        landing_x = np.random.uniform(x_min, x_max)
        landing_y = np.random.uniform(y_min, y_max)
        configuration = np.array([approach_radius, approach_angle, landing_x, landing_y])

        return configuration

    def optimize_plan(self, idx: Optional[int] = None):

        # sample initial guess
        initial_simplex = []
        for i in range(5):
            initial_simplex.append(self.sample_configuration())
        initial_simplex = np.array(initial_simplex)

        # set optimization bounds
        landing_target_aabb = self.landing_target.get_AABB().cpu().numpy()
        x_min = landing_target_aabb[0][0]
        x_max = landing_target_aabb[1][0]
        y_min = landing_target_aabb[0][1]
        y_max = landing_target_aabb[1][1]

        radii_bounds = np.array([-np.inf, np.inf])
        angles_bounds = np.array([-np.inf, np.inf])
        x_bounds = np.array([x_min, x_max])
        y_bounds = np.array([y_min, y_max])


        opt_solution, opt_loss, trajectories, f_trajectories = nelder_mead(
            self.evaluate_rollouts,
            x_starts=np.array([initial_simplex[0]]),
            initial_simplexes=initial_simplex[None, :, :],
            scale_factors=np.array([0.5, 0.5, 0.25, 0.25]),
            bounds = np.array([radii_bounds, angles_bounds, x_bounds, y_bounds]),
        )

        # create directory
        run_name = self.config["dataset_config"]["dataset_path"].split("/")[-1] 
        planning_directory = Path(self.config["save_directory"]) / run_name / "planning"
        create_directory(planning_directory, overwrite=False)

        # package optimal action
        try:
            # action_dict = {
            #     "approach_radius": opt_solution[0],
            #     "approach_angle": opt_solution[1],
            #     "landing_x": opt_solution[2],
            #     "landing_y": opt_solution[3],
            #     "flight_altitude": self.flight_altitude,
            #     "initial_position": self.robot_info["initial_position"],
            #     "velocity": self.robot_info["approach_velocity"],
            # }

            approach_radius = opt_solution[0]
            approach_angle = opt_solution[1]
            landing_x = opt_solution[2]
            landing_y = opt_solution[3]
            landing_position = np.array([landing_x, landing_y])
            approach_waypoint = landing_position + approach_radius * np.array([np.cos(approach_angle), np.sin(approach_angle)])

            action_dict = {
                "initial_position": self.robot_info["initial_position"][:2].tolist(),
                "approach_waypoint": approach_waypoint[:2].tolist(),
                "landing_position": landing_position[:2].tolist(),
                "landing_target_height": (self.flight_altitude - self.landing_height).tolist(),
                "flight_altitude": self.flight_altitude,
                "flight_velocity": self.robot_info["approach_velocity"],
            }
            if idx is not None:
                save_dict_to_json(action_dict, f"action_specification_{idx}.json", directory=planning_directory)
            else:
                save_dict_to_json(action_dict, "action_specification.json", directory=planning_directory)
        except Exception as e:
            print("Error saving action specification: ", e)
            breakpoint()
    
    def optimization_sweep(self):
        # load initial positions from baseline reference data
        dataset_path = Path(self.config["dataset_config"]["dataset_path"])
        baseline_data_path = dataset_path / "baseline"
        poses = np.load(str(baseline_data_path / "camera_to_world_transforms.npy"))

        for i in range(len(poses) - 1):
            initial_position = poses[i].reshape(4, 4)[:3, 3]
            self.robot_info["initial_position"] = initial_position
            self.optimize_plan(idx=i)





        # test_configuration = np.array([0.53598723,  0.27573707, -0.19303532, -0.0202171])
        # self.evaluate_rollouts(test_configuration)


        # losses = self.evaluate_rollouts(configurations)
        # losses[losses > 5] = 1.5

        # # plot losses and corresponding configurations in 3D
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(configurations[:, 0], configurations[:, 1], configurations[:, 2], c=losses, cmap='viridis')
        # ax.set_xlabel('X Position')
        # ax.set_ylabel('Y Position')
        # ax.set_zlabel('Z Position')
        # ax.set_title('Losses for Different Configurations')
        # plt.colorbar(ax.scatter(configurations[:, 0], configurations[:, 1], configurations[:, 2], c=losses, cmap='viridis'))
        # plt.show()

        # breakpoint()

        # # save results
        # # create directory
        # run_name = self.config["dataset_config"]["dataset_path"].split("/")[-1] 
        # planning_directory = Path(self.config["save_directory"]) / run_name / "landing_site_planning"
        # create_directory(planning_directory, overwrite=True)
        
        # # save results
        # np.save(str(planning_directory) + "/landing_sites.npy", configurations)
        # np.save(str(planning_directory) + "/landing_site_losses.npy", losses)


    
    
        




    


