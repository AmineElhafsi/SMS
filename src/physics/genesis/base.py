from pathlib import Path
from typing import Dict, Optional, Union

import genesis as gs
import json
import numpy as np
import trimesh

from src.physics.sim_entities import load_sim_object, Sphere


class BaseEnv:
    def __init__(
        self,
        env_name: str,
        config: Dict,
    ) -> None:
        
        self.env_name = env_name
        self.config = config

        # set up simulation scene
        physics_config = config["physics_simulation"]
        
        if physics_config["backend"] == "gpu":
            backend = gs.gpu
        elif physics_config["backend"] == "cpu":
            backend = gs.cpu
        else:
            raise ValueError(f"Unknown backend: {physics_config['backend']}")

        if not gs._initialized:
            gs.init(backend=backend, logging_level=physics_config["logging_level"])

        physics_config["rigid_options"]["integrator"] = gs.integrator.Euler
        physics_config["rigid_options"]["constraint_solver"] = gs.constraint_solver.Newton
        physics_config["rigid_options"]["max_collision_pairs"] = 10000
        physics_config["rigid_options"]["ls_tolerance"] = 0.001
        physics_config["rigid_options"]["box_box_detection"] = True

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                **physics_config["sim_options"]
            ),
            rigid_options=gs.options.RigidOptions(
                **physics_config["rigid_options"]
            ),
            viewer_options = gs.options.ViewerOptions(
                max_FPS = 1. / physics_config["sim_options"]["dt"],
            ),
            show_viewer = physics_config["show_viewer"],
        )
        self.sim_duration = physics_config["sim_duration"]
        self.sim_steps = int(self.sim_duration / physics_config["sim_options"]["dt"])

        # add camera
        self.cam = self.scene.add_camera(
            res    = (1280, 800),
            pos    = (2.5, 2.5, 1.5),
            lookat = (0, 0, 0.5),
            fov    = 70,
            GUI    = True
        )

        # populate the scene with entities
        self.sim_entities = {}
        self.entity_masses = {}
        self._populate_scene()

        # place holders for robot and controllers
        self.robot = None
        self.robot_info = None
        self.controllers = None
        
    def build(self):
        # self.scene.build(n_envs=self.config["physics_simulation"]["n_envs"], env_spacing=(5.0, 5.0))
        self.scene.build()

    def add_ground_plane(self, friction: float = 0.01, restitution: float = 0.6): #friction: float = 0.66, restitution: float = 0.6):
        ground_plane = self.scene.add_entity(
            gs.morphs.Plane(collision=True),
            material=gs.materials.Rigid(friction=friction, coup_restitution=restitution),
        )
        self.sim_entities["ground_plane"] = ground_plane

    def add_entity(self, entity_dir: Union[str, Path]):
        entity_dir = Path(entity_dir) if isinstance(entity_dir, str) else entity_dir

        # load metadata json
        metadata_file = entity_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                entity_info = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found in {entity_dir}")
        
        # load material file
        material_file = entity_dir / "materials.json"
        if material_file.exists():
            with open(material_file, "r") as f:
                material_dict = json.load(f)
            material = gs.materials.Rigid(
                rho=material_dict["Density"], 
                friction=material_dict["Friction Coefficient"], 
                coup_restitution=material_dict["Coefficient of Restitution"],
            )
        else:
            raise FileNotFoundError(f"Material file not found in {entity_dir}")
        
        # Find mesh file with either .npz or .obj extension
        mesh_file = entity_dir / "mesh_aligned.obj"

        entity = self.scene.add_entity(
            gs.morphs.Mesh(
                file=str(mesh_file),
                convexify=True,
                decimate=True,
                decimate_aggressiveness=0,
                # coacd_options=gs.options.CoacdOptions(
                #     threshold=0.05
                # ),
                # decompose_nonconvex=False,
            ),
            material=material,
            # vis_mode="collision",
        )

        
        # mesh_file = None
        # for ext in ['.npz', '.obj']:
        #     candidate_file = entity_dir / f"mesh{ext}"
        #     if candidate_file.exists():
        #         mesh_file = candidate_file
        #         break

        # if mesh_file is None:
        #     raise FileNotFoundError(f"Mesh file not found in {entity_dir}")

        # # add entity to sim
        # if mesh_file.suffix == ".npz":
        #     sim_object = load_sim_object(mesh_file)
        #     if sim_object.entity_type == "sphere":
        #         entity = self.scene.add_entity(
        #                 gs.morphs.Sphere(
        #                     pos=sim_object.center,
        #                     radius=sim_object.radius,
        #                 ),
        #                 material=material,
        #             )
        #     else:
        #         raise NotImplementedError(f"Unsupported entity type: {sim_object.entity_type}")
        # elif mesh_file.suffix == ".obj":
            # entity = self.scene.add_entity(
            #         gs.morphs.Mesh(
            #             file=str(mesh_file),
            #         ),
            #         material=material,
            #     )
        
        entity_key = "_".join((entity_info["label"], str(entity_info["class_id"])))
        self.sim_entities[entity_key] = entity

        if any(m in material_dict["Material"].lower() for m in ["plastic", "cardboard"]):
            mesh = trimesh.load(mesh_file)
            area = mesh.area
            if material_dict["Material"].lower() == "cardboard":
                thickness = 0.001
            elif material_dict["Material"].lower() == "plastic":
                thickness = 0.006
            else:
                raise
            volume = area * thickness
            mass = volume * material_dict["Density"]
            self.entity_masses[entity_key] = mass
        print("Entity masses: ", self.entity_masses)
        
    def _populate_scene(self, include_ground: bool = True) -> None:
        sim_entity_dict = {}

        # create ground plane
        if include_ground:
            ground_plane = self.add_ground_plane()

        # introduce scanned objects
        run_name = self.config["dataset_config"]["dataset_path"].split("/")[-1]
        sim_entities_directory = Path(self.config["save_directory"]) / run_name / "entities"
        for sim_entity_path in sim_entities_directory.iterdir():
            # skip ground and background
            if any(keyword in sim_entity_path.name for keyword in ["ground", "background"]):
                continue
            
            print("Adding entity: ", sim_entity_path.name)
            entity = self.add_entity(sim_entity_path)

    # def _populate_scene(self, include_ground: bool = True) -> None:
    #     sim_entity_dict = {}

    #     # create ground plane
    #     if include_ground:
    #         ground_plane = self.add_ground_plane()


    #     color_mapping = {
    #         "plastic": (0.83, 0.83, 0.83, 1.0),  # Light gray
    #         "cardboard": (0.76, 0.60, 0.42, 1.0),  # Brownish
    #         "aluminum": (0.75, 0.75, 0.75, 1.0),  # Silver
    #         "wood": (0.54, 0.27, 0.07, 1.0),  # Brown
    #         "steel": (0.5, 0.5, 0.5, 1.0),  # Dark gray
    #     }

    #     tower_base = self.scene.add_entity(
    #         gs.morphs.Mesh(
    #             file="/home/anonymous/Documents/Research/gaussian-splatting-playground/data/runs/flight_room_scene_1/entities/a plastic recycling bin_class_id_0/optimized_mesh_aligned.obj",
    #         ),
    #         material=gs.materials.Rigid(
    #             rho=700/42,
    #         ),
    #         surface=gs.surfaces.Default(
    #             color=(0.83, 0.83, 0.83, 1.0),  # A light gray color
    #         ),
    #     )
    #     tower_roof = self.scene.add_entity(
    #         gs.morphs.Mesh(
    #             file=str("/home/anonymous/Documents/Research/gaussian-splatting-playground/data/runs/flight_room_scene_1/entities/a cardboard box_class_id_1/optimized_mesh_aligned.obj"),
    #             pos=(0., 0., 0.0),
    #         ),
    #         material=gs.materials.Rigid(
    #             rho=700/121,
    #         ),
    #         surface=gs.surfaces.Default(
    #             color=(0.76, 0.60, 0.42, 1.0),  # A brownish color to represent cardboard
    #         ),
    #     )
    #     self.sim_entities["tower_base"] = tower_base
    #     self.sim_entities["tower_roof"] = tower_roof

    def simulate(self, num_steps: Optional[int] = None) -> None:
        robot_positions = []
        if num_steps is None:
            # if self.controllers is not None:
            self.apply_control()
            self.scene.step()
        else:
            for idx in range(num_steps):
                if idx % 100 == 0:
                    breakpoint()
                # if self.controllers is not None:
                self.apply_control()
                self.scene.step()
                robot_positions.append(self.robot.get_links_pos())

    def evaluate_rollouts(self, configurations: np.ndarray) -> np.ndarray:
        # dimension wrangling
        if len(configurations.shape) == 1:
            configurations = np.expand_dims(configurations, axis=0)
        n_configurations = len(configurations)

        # if there are more configurations than environments, we need to batch the calls
        if n_configurations > self.scene.n_envs:
            inc = max(self.scene.n_envs, 1)
            configurations = [configurations[i:i + inc] for i in range(0, len(configurations), inc)]
        else:
            configurations = [configurations]

        # evaluate each batch of configurations and collect losses
        losses = []
        for configuration_batch in configurations:
            self.apply_configurations(configuration_batch)
            self.simulate(num_steps=self.sim_steps)
            batch_losses = self.compute_task_loss()[:n_configurations]
            losses.append(batch_losses)
            self.reset()
        losses = np.concatenate(losses)

        return losses

    def parse_task(self):
        raise NotImplementedError
    
    def add_robot(self):
        raise NotImplementedError
    
    def add_controller(self):
        raise NotImplementedError
    
    def get_initial_configuration(self):
        raise NotImplementedError
    
    def generate_configuration_grid(self):
        raise NotImplementedError
    
    def apply_configurations(self, configurations: np.ndarray) -> None:
        raise NotImplementedError
    
    def compute_task_loss(self):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
