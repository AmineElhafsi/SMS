from pathlib import Path
from typing import Dict, Optional, Union

import json
import numpy as np
import pybullet as p 
import pybullet_data

from src.physics.sim_entities import load_sim_object, Sphere
from src.utils.meshing import compute_multibody_mesh_volume

import numpy as np
import trimesh


class BaseEnv:
    def __init__(
        self,
        env_name: str,
        config: Dict,
        # physics_config: Dict,
        # task_config: Dict,
    ) -> None:
        
        self.env_name = env_name
        self.config = config
        
        
        # sim parameters
        physics_config = config["physics_simulation"]
        self.sim_duration = physics_config["sim_duration"]
        self.dt = physics_config["sim_options"]["dt"]
        self.sim_steps = int(self.sim_duration / physics_config["sim_options"]["dt"])
        self.n_envs = physics_config["n_envs"]

        # connect to PyBullet
        p.connect(
            p.GUI, 
            options="--background_color_red=1 --background_color_blue=1 --background_color_green=1"
        )
        print("Connected to PyBullet")
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        p.setTimeStep(self.dt)
        p.setGravity(0., 0., -9.81)
        # p.setPhysicsEngineParameter(numSolverIterations=50)  # Increase solver iterations
        # p.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSubSteps=4)  # Add substeps
        # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)
        # p.configureDebugVisualizer(p.COV_ENABLE_AABB, 1)
        # p.configureDebugVisualizer(p.COV_ENABLE_CONTACT_POINTS, 1)

        if physics_config["show_viewer"]:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=0,
                cameraPitch=-40,
                cameraTargetPosition=[0, 0, 0],
            )

        # populate the scene with entities
        self.sim_entities = {}
        self._populate_scene()

        # configure robot and controllers
        self.robot_id = None
        self.robot_info = None
        self.controllers = None
        self.use_robot = config["scenario"]["settings"]["use_robot"]
        if self.use_robot:
            self.add_robot()
            self.add_controllers()

        # parse task
        self.parse_task()

        # reset / initialize environment    
        self.sim_reset_state = None
        self.reset()

    def add_ground_plane(self, friction: float = 0.66, restitution: float = 0.6): #friction: float = 0.66, restitution: float = 0.6):
        # ground_collision_shape = p.createCollisionShape(p.GEOM_PLANE)
        # ground_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_collision_shape)
        # p.changeVisualShape(ground_id, -1, rgbaColor=[1, 1, 1, 1])
        ground_id = p.loadURDF("plane.urdf")
        p.changeDynamics(ground_id, -1, lateralFriction=friction, restitution=restitution)
        self.sim_entities["ground_plane"] = ground_id

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
        else:
            raise FileNotFoundError(f"Material file not found in {entity_dir}")
        
        # Find mesh file with either .npz or .obj extension
        mesh_file = None
        for ext in ['.npz', '.obj']:
            candidate_file = entity_dir / f"mesh{ext}"
            if candidate_file.exists():
                mesh_file = candidate_file
                break

        if mesh_file is None:
            raise FileNotFoundError(f"Mesh file not found in {entity_dir}")

        # add entity to sim
        if mesh_file.suffix == ".npz":
            sim_object = load_sim_object(mesh_file)
            if sim_object.entity_type == "sphere":
                # get mass
                volume = 4 / 3 * np.pi * sim_object.radius ** 3
                mass = material_dict["Density"] * volume

                # create object
                collision_shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=sim_object.radius)
                entity_id = p.createMultiBody(
                    baseMass=mass,
                    baseCollisionShapeIndex=collision_shape_id,
                    baseVisualShapeIndex=collision_shape_id,
                    basePosition=sim_object.center
                )
                
                # set material properties
                p.changeDynamics(
                    entity_id, 
                    -1, 
                    lateralFriction=material_dict["Friction Coefficient"],
                    restitution=material_dict["Coefficient of Restitution"],
                    linearDamping=0,
                    angularDamping=0.05,
                )

                print("Object Name: ", str(entity_dir))
                print("Radius: ", sim_object.radius)
                print("Center: ", sim_object.center)

            else:
                raise NotImplementedError(f"Unsupported entity type: {sim_object.entity_type}")
        elif mesh_file.suffix == ".obj":
            decomposed_mesh_file = entity_dir / "mesh_decomposed.obj"

            # get mass
            # volume = compute_multibody_mesh_volume(decomposed_mesh_file)
            # mass = material_dict["Density"] * volume
            # print("Mass: ", mass)

            # create object
            mesh = trimesh.load_mesh(decomposed_mesh_file)
            mesh.density = material_dict["Density"]
            com = mesh.center_mass
            mass = mesh.mass

            T = mesh.principal_inertia_transform
            inertia_rotation = trimesh.transformations.quaternion_from_matrix(T[:3, :3].T)
            inertia_rotation = np.roll(inertia_rotation, -1)

            collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=str(decomposed_mesh_file))
            entity_id = p.createMultiBody(
                baseMass=mass,
                baseInertialFramePosition=com,
                baseInertialFrameOrientation=inertia_rotation,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=collision_shape_id,
            )
                        
            # set material properties
            p.changeDynamics(
                entity_id, 
                -1,
                localInertiaDiagonal=mesh.principal_inertia_components,
                lateralFriction=material_dict["Friction Coefficient"],
                restitution=material_dict["Coefficient of Restitution"],
                linearDamping=0,
                angularDamping=0.05,
            )
        
        entity_key = "_".join((entity_info["label"], str(entity_info["class_id"])))
        self.sim_entities[entity_key] = entity_id
        
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
            
            entity = self.add_entity(sim_entity_path)

    def simulate(self, num_steps: Optional[int] = None) -> None:
        # camera
        # from PIL import Image

        # camera_pos = [1, 2, 3]           # Camera position (x, y, z)
        # target_pos = [0, 0, 0]           # Where the camera looks at
        # up_vector = [0, 0, 1]            # Up direction

        # fov = 40                         # Field of view in degrees
        # aspect = 1.0                     # Aspect ratio (width/height)
        # near = 0.1                       # Near clipping plane
        # far = 100                        # Far clipping plane
        # width = 1280                      # Image width
        # height = 800                     # Image height

        # view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
        # proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        


        import time
        if num_steps is None:
            if self.controllers is not None:
                self.apply_control()
                self.log_simulation_data()
            p.stepSimulation(self.dt)
        else:
            for t in range(num_steps):
                # if t % 50 == 0:
                #     img = p.getCameraImage(width, height, view_matrix, proj_matrix, shadow=1, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                #     rgb_array = np.reshape(img[2], (height, width, 4))  # RGBA
                #     rgb_img = rgb_array[:, :, :3]  # Drop alpha if not needed
                #     im = Image.fromarray(rgb_img.astype(np.uint8))
                #     segmentation_mask = np.reshape(img[4], (height, width))
                #     im.save(f"./virtual_environment_frames/{t:04d}.png")
                #     np.save(f"./virtual_environment_frames/{t:04d}_seg.npy", segmentation_mask)


                if self.controllers is not None:
                    self.apply_control()
                    self.log_simulation_data()
                    # if self.config["physics_simulation"]["show_viewer"]:
                    #     time.sleep(self.dt*4)
                p.stepSimulation()

    def evaluate_rollouts(self, configurations: np.ndarray, reset: bool = True) -> np.ndarray:
        # dimension wrangling
        if len(configurations.shape) == 1:
            configurations = np.expand_dims(configurations, axis=0)
        n_configurations = len(configurations)

        # if there are more configurations than environments, we need to batch the calls
        if n_configurations > self.n_envs:
            configurations = np.array_split(configurations, n_configurations // self.n_envs)
        else:
            configurations = [configurations]

        # evaluate each batch of configurations and collect losses
        losses = []
        for configuration_batch in configurations:
            self.apply_configurations(configuration_batch)
            self.simulate(num_steps=self.sim_steps)
            batch_losses = self.compute_task_loss()[:n_configurations]
            losses.append(batch_losses)
            if reset:
                self.reset()
        losses = np.concatenate(losses)

        return losses
    
    def add_robot(self):
        raise NotImplementedError
    
    def add_controllers(self):
        raise NotImplementedError
    
    def apply_configurations(self, configurations: np.ndarray) -> None:
        raise NotImplementedError
    
    def generate_configuration_grid(self):
        raise NotImplementedError
    
    def get_initial_configuration(self):
        raise NotImplementedError    
    
    def compute_task_loss(self):
        raise NotImplementedError
    
    def log_simulation_data(self):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError
    
    def parse_task(self):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
