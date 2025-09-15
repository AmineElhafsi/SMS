import os

from datetime import datetime
from pathlib import Path
import shutil
from typing import Union

import numpy as np
import torch

from src.splatting.datasets import get_dataset
from src.splatting.gaussian_model import GaussianModel
from src.splatting.data_logging import Logger
from src.splatting.mapper import Mapper
from src.splatting.parameters import OptimizationParams
from src.utils.camera import exceeds_motion_thresholds
from src.utils.io import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.rgbd import depth_to_pointcloud, get_normals
from src.utils.utils import setup_seed


class SplattingManager:
    def __init__(self, mode) -> None:
        # index tracking
        if mode == "submap":
            self.submap_id = 0 # submap index
            self.new_submap_frame_ids = [0] # keyframe indices where new submap was started
        
        # camera tracking
        self.current_keyframe = None # current keyframe
        self.keyframes = [] # keyframes
        # self.results = {} # info output after frame optimization


    def get_current_frame_id(self) -> int:
        return self.current_keyframe["frame_id"]
    
    def get_current_pose(self) -> np.ndarray:
        return self.current_keyframe["T_c2w"]
    
    def get_submap_reference_pose(self) -> np.ndarray:
        return self.keyframes[self.new_submap_frame_ids[-1]]["T_c2w"]

    def log_keyframe(self, keyframe: dict) -> None:
        self.current_keyframe = keyframe
        self.keyframes.append(keyframe)

    def save_keyframes(self, output_directory: Union[str, Path]) -> None:
        save_dict_to_ckpt(self.keyframes, 
            "keyframes.ckpt", 
            directory=output_directory
        )

    def save_submap(self, gaussian_model: GaussianModel, output_directory: Union[str, Path]) -> None:
        # retrieve current frame id
        frame_id = self.get_current_frame_id()

        # save current submap and camera trajectory
        gaussian_params = gaussian_model.capture_dict()
        submap_ckpt = {
            "gaussian_params": gaussian_params,
            "submap_keyframes": list(range(self.new_submap_frame_ids[-1], frame_id)),
        }
        save_dict_to_ckpt(
            submap_ckpt,
            f"submap_{self.submap_id}".zfill(6) + ".ckpt",
            directory=(output_directory/"maps")
        )

    def save_map(self, gaussian_model: GaussianModel, output_directory: Union[str, Path]) -> None:
        # retrieve current frame id
        frame_id = self.get_current_frame_id()

        # save current submap and camera trajectory
        gaussian_params = gaussian_model.capture_dict()
        map_ckpt = {
            "gaussian_params": gaussian_params
        }
        save_dict_to_ckpt(
            map_ckpt,
            "map.ckpt",
            directory=(output_directory/"maps")
        )
        
    def start_new_submap(self):
        # retrieve current frame id
        frame_id = self.get_current_frame_id()

        # update submap tracking
        self.submap_id += 1
        self.new_submap_frame_ids.append(frame_id)
                

class GaussianSplatting:
    def __init__(self, config: dict) -> None:
        # prepare output
        # self._setup_output_directory(config)
        data_name = config["dataset_config"]["dataset_path"].split("/")[-1] 
        self.output_directory = Path(config["save_directory"]) / data_name
        save_dict_to_yaml(config, "config.yaml", directory=self.output_directory)

        # parse configuration
        self.config = config
        self.mode = config["mode"]
        self.submap_creation_criterion = config["mapping"]["submap_creation_criteria"]["criterion"]
        self.gaussian_model_config = config["gaussian_model"]
        self.include_features = config["gaussian_model"]["include_point_features"]

        # prepare logging
        self.logger = Logger(self.output_directory, use_wandb=False)
        
        # initialize model
        setup_seed(self.config["seed"])
        self.opt = OptimizationParams()

        self.gaussian_model = GaussianModel(**self.gaussian_model_config)
        self.gaussian_model.training_setup(self.opt)        

        # prepare splatting management / tracking
        self.manager = SplattingManager(self.mode)

        # mapping module
        self.mapper = Mapper(config["mapper"], self.logger)

    # def _setup_output_directory(self, config: dict) -> None:
    #     """ 
    #     Sets up the output path for saving results based on the provided configuration. If the output path is not
    #     specified in the configuration, it creates a new directory.

    #     Args:
    #         config: A dictionary containing the experiment configuration including data and output path information.
    #     """
    #     data_name = config["dataset_config"]["dataset_path"].split("/")[-1] 
    #     self.output_directory = Path(config["save_directory"]) / data_name

    #     # check if output directory already exists
    #     if self.output_directory.exists():
    #         if not config["overwrite"]:
    #             raise ValueError(f"Output directory {self.output_directory} already exists.")
    #         else:
    #             print(f"Output directory {self.output_directory} already exists. Overwriting.")
    #             # delete existing directory and create a new one
    #             shutil.rmtree(self.output_directory)
    #             self.output_directory.mkdir(parents=True)
    #             os.makedirs(self.output_directory / "mapping", exist_ok=True)
    #     else:
    #         self.output_directory.mkdir(parents=True)
    #         os.makedirs(self.output_directory / "mapping", exist_ok=True)

    def should_start_new_submap(self) -> bool:
        if self.submap_creation_criterion == "motion_threshold":
            start_new_map = exceeds_motion_thresholds(
                torch.from_numpy(self.manager.get_current_pose()),
                torch.from_numpy(self.manager.get_submap_reference_pose()),
                self.config["mapping"]["submap_creation_criteria"]["translation_threshold"],
                self.config["mapping"]["submap_creation_criteria"]["rotation_threshold"]
            )
            return start_new_map
        else:
            raise NotImplementedError(f"Criterion {self.submap_creation_criterion} not implemented.")

    def start_new_submap(self) -> None:
        """ 
        Initializes a new submap, saving the current submap's checkpoint and resetting the Gaussian model.
        This function updates the submap count and optionally marks the current frame ID for new submap initiation.
        """
        # save current submap and all keyframes to present
        self.manager.save_submap(self.gaussian_model, self.output_directory)
        self.manager.save_keyframes(self.output_directory)
        self.manager.start_new_submap()

        # reset Gaussian model and keyframe info
        self.gaussian_model = self.GaussianModel(**self.gaussian_model_config)
        self.mapper.keyframes = []

    def step(self, keyframe: dict, init: bool) -> None:
        # start new submap if needed
        is_new_submap = init #True if keyframe["frame_id"] == 0 else False # set true by default only for first iteration
        if self.mode == "submap" and self.should_start_new_submap():
            print("Starting new submap")
            self.start_new_submap()
            is_new_submap = True

        # map frame
        print(f"Mapping frame {self.manager.get_current_frame_id()}")
        # self.gaussian_model.training_setup(self.opt)
        results_dict = self.mapper.map(
            keyframe,
            self.gaussian_model,
            is_new_submap
        )

    def run_from_dataset(self, dataset: torch.utils.data.Dataset) -> None:
        # temporary
        num_classes = len(dataset[0][5])
        if self.config["mapper"]["affinity_features"]["optimize"] and self.config["mapper"]["affinity_features"]["method"] == "discriminative":
            self.gaussian_model.point_classifier_setup(num_classes=num_classes)

        # iterate over dataset
        start_frame = 0
        end_frame = len(dataset)
        frame_skip = 1

        if "start_frame" in self.config["mapping"] and self.config["mapping"]["start_frame"] > 0:
            start_frame = self.config["mapping"]["start_frame"]
        if "end_frame" in self.config["mapping"] and self.config["mapping"]["end_frame"] > start_frame:
            end_frame = self.config["mapping"]["end_frame"]
        if "frame_skip" in self.config["mapping"] and self.config["mapping"]["frame_skip"] > 1:
            frame_skip = self.config["mapping"]["frame_skip"]

        frame_indices = list(range(start_frame, end_frame, frame_skip))

        for frame_id in frame_indices:
            # get keyframe from dataset
            keyframe = {
                "frame_id": frame_id,
                "T_c2w": dataset[frame_id][1],
                "color": dataset[frame_id][2],
                "depth": dataset[frame_id][3],
                "segmentation": dataset[frame_id][4],
                "segmentation_annotation": dataset[frame_id][5],
                "normals": get_normals(dataset[frame_id][3], dataset.K, dataset[frame_id][1]),    
                "H": dataset.image_height,
                "W": dataset.image_width,
                "K": dataset.K, # camera intrinsics matrix
            }

            # log keyframe (without image/depth data to minimize storage)
            self.manager.log_keyframe(
                {
                    "frame_id": frame_id,
                    "T_c2w": dataset[frame_id][1],
                    "H": dataset.image_height,
                    "W": dataset.image_width,
                    "K": dataset.K,
                }
            )

            # step
            # if first loop set init to true
            init = True if frame_id == start_frame else False
            self.step(keyframe, init)

        # # save data before terminating
        # if self.mode == "submap":
        # # save current submap and all keyframes to present
        #     self.manager.save_submap(self.gaussian_model, self.output_directory)
        #     self.manager.save_keyframes(self.output_directory)
        try:
            self.manager.save_map(self.gaussian_model, self.output_directory)
        except:
            breakpoint()

        return self.gaussian_model

if __name__ == "__main__":
    import yaml

    # load yaml file config
    with open("/home/anonymous/Documents/Research/gaussian-splatting-playground/config/test.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # Create an instance of GaussianSplatting with the loaded config
    splatting = GaussianSplatting(config)
    
    # Call the run_from_dataset method with the desired dataset
    splatting.run_from_dataset()