import copy
import shutil

from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pybullet as p
import pytorch3d.ops
import torch
import torch.nn.functional as F

from src.data.preprocessing import DataPreprocessor
from src.physics.pybullet.billiards import Billiards
from src.physics.genesis.quadrotor import QuadrotorLanding
from src.splatting.datasets import get_dataset
from src.splatting.gaussian_model import GaussianModel
from src.splatting.gaussian_splatting import GaussianSplatting
from src.utils.camera import depth_image_to_pointcloud
from src.utils.geometry_3d import truncate_point_cloud, trim_point_cloud_edges, point_cloud_to_triangle_mesh
from src.utils.io import create_directory, get_unique_file_path, save_dict_to_ckpt
# from src.utils.rendering import get_render_settings, render_gaussian_model, render_gaussian_model_features
# from src.utils.segmentation import perform_entity_discovery, merge_duplicate_entities, segment_point_cloud, knn_point_cloud_smoothing
# from src.utils.splatting import sample_control_points
# from src.utils.utils import numpy_to_torch
# from src.vision_models.detection_models import OWLv2
# from src.vision_models.segmentation_models import SAM2
# from src.vision_models.utils.detection import adjust_bboxes, inflate_bboxes, get_image_foreground, show_detections
# from src.vision_models.utils.segmentation import get_background_mask, get_ground_mask, fill_holes, show_box, show_mask

# import pyransac3d as pyrsc
from src.physics.sim_entities import Mesh, Sphere
from src.utils.geometry_3d import densify_point_cloud
from src.utils.math_utils import rgb_to_sh
from src.utils.splatting import refine_gaussian_model_segmentation


class ExperimentRunner:
    def __init__(self, config: dict):
        self.config = config

        dataset_config = config["dataset_config"]
        self.dataset = get_dataset(dataset_config["dataset"])(dataset_config)

    def run_data_preprocessing(self):
        data_preprocessor = DataPreprocessor(self.config["preprocessing"], self.config["dataset_config"])
        data_preprocessor.run()

        # hacky, but reload the dataset to include the new instance segmentation masks
        dataset_config = self.config["dataset_config"]
        self.dataset = get_dataset(dataset_config["dataset"])(dataset_config)

    def run_splatting(self):
        # map scene
        splatting = GaussianSplatting(self.config)
        gaussian_model = splatting.run_from_dataset(self.dataset)

    def run_materialization(self):
        from src.materialization.materializer import Materializer
        
        # get saved gaussian model
        run_name = self.config["dataset_config"]["dataset_path"].split("/")[-1] 
        gaussian_model_path = Path(self.config["save_directory"]) / run_name / "maps" / "map.ckpt"
        gaussian_parameter_dict = torch.load(gaussian_model_path)["gaussian_params"]
        gaussian_model = GaussianModel()
        gaussian_model.load_parameters(gaussian_parameter_dict)

        materializer = Materializer(self.config, gaussian_model)
        materializer.run()
        materializer.mesh_alignment()

    def run_virtual_sim_planner(self):
        
        if self.config["scenario"]["type"] == "billiards":
            physics_env = Billiards(self.config)
            physics_env.optimize_plan()
            p.disconnect()
        elif self.config["scenario"]["type"] == "quadrotor":
            physics_env = QuadrotorLanding(self.config)
            physics_env.optimize_plan()
        else:
            raise NotImplementedError(f"Scenario type {self.config['scenario']['type']} not implemented.")

    def run_optimization_sweep(self):
        physics_env = QuadrotorLanding(self.config)
        physics_env.optimization_sweep()

    def eval_virtual_sim_plan(self):
        physics_env = Billiards(self.config)

        # get optimized action
        run_name = self.config["dataset_config"]["dataset_path"].split("/")[-1] 
        planning_directory = Path(self.config["save_directory"]) / run_name / "planning"

        # load json file
        import json

        # with open(str(planning_directory) + "/action_specification", 'r') as stream:
        #     action_specification = json.load(stream)

        with open("/home/anonymous/Documents/Research/gaussian-splatting-playground/data/runs/billiards_scene_10/planning_results/action_specification_14.json", 'r') as stream:
            action_specification = json.load(stream)

        configurations = np.array(
            [action_specification["contact_speed"], action_specification["contact_angle"], action_specification["contact_angle"]]
        )
        physics_env.evaluate_rollouts(
            configurations
        )


    



    