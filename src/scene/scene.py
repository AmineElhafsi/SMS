from pathlib import Path
from typing import Union

import open3d as o3d
import numpy as np
import torch

from plyfile import PlyData, PlyElement

from src.splatting.gaussian_model import GaussianModel
from src.scene.utils.scene_processing import merge_submaps, refine_global_map
from src.utils.camera import get_camera_intrinsics_matrix
from src.utils.io import load_yaml
from src.utils.utils import numpy_to_point_cloud, torch_to_numpy


class Scene:
    def __init__(self, config: dict, scene_data_path: Union[Path, str] = None) -> None:
        self.config = config
        self.scene_data_path = Path(scene_data_path) if scene_data_path else None

        # camera
        self.K = get_camera_intrinsics_matrix(config)

        # prepare scene
        self.submaps = None
        self.training_keyframes = None

    @classmethod
    def from_directory(cls, scene_data_directory: Union[Path, str]) -> "Scene":
        scene_data_directory = Path(scene_data_directory) if isinstance(scene_data_directory, str) else scene_data_directory

        config = load_yaml(scene_data_directory / "config.yaml")
        scene = cls(config, scene_data_directory)
        scene.load_data(scene_data_directory)
        return scene
    
    def assemble_global_map(self):
        assert self.submaps is not None, "No submaps loaded."
        merged_point_cloud = merge_submaps(self.submaps)
        gaussian_model = refine_global_map(
            merged_point_cloud, 
            self.training_keyframes, 
            0.005, 
            refinement_iterations=10000
        )
        return gaussian_model

    def load_data(self, scene_data_directory: Union[Path, str]):
        submap_directory = scene_data_directory / "submaps"
        submap_paths = list(submap_directory.glob("submap_*.ckpt"))
        submap_paths.sort(key=lambda path: int(path.stem.split('_')[-1]))

        keyframe_directory = scene_data_directory / "keyframes"
        keyframe_paths = list(keyframe_directory.glob("keyframe_*.ckpt"))
        keyframe_paths.sort(key=lambda path: int(path.stem.split('_')[-1]))

        self.submaps = []
        for p in submap_paths:
            self.submaps.append(torch.load(p))

        self.training_keyframes = []
        for p in keyframe_paths:
            self.training_keyframes.append(torch.load(p))

    def save_global_map(self, gaussian_model: GaussianModel, path: Union[Path, str] = None):
        path = Path(path) if path else self.scene_data_path / "global_map.ply"
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        xyz = gaussian_model._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            gaussian_model._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy())
        f_rest = (
            gaussian_model._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy())
        opacities = gaussian_model._opacity.detach().cpu().numpy()
        if gaussian_model.isotropic:
            # tile into shape (P, 3)
            scale = np.tile(gaussian_model._scaling.detach().cpu().numpy()[:, 0].reshape(-1, 1), (1, 3))
        else:
            scale = gaussian_model._scaling.detach().cpu().numpy()
        rotation = gaussian_model._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in gaussian_model.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def visualize_submap(self, submap_idx):
        # submap = self.submaps[submap_idx]
        # gaussian_params = submap["gaussian_params"]
        # submap_pts = torch_to_numpy(gaussian_params["xyz"])
        # point_cloud = numpy_to_point_cloud(submap_pts, np.zeros_like(submap_pts))

        # o3d.visualization.draw_geometries([point_cloud])

        # submap = self.submaps[0]
        # gaussian_params = submap["gaussian_params"]
        # submap_pts = torch_to_numpy(gaussian_params["xyz"])
        # rgb = np.zeros_like(submap_pts)
        # rgb[:, 0] = 255
        # point_cloud1 = numpy_to_point_cloud(submap_pts, rgb)
        # point_cloud1 = point_cloud1.voxel_down_sample(voxel_size=0.05)

        submap = self.submaps[1]
        gaussian_params = submap["gaussian_params"]
        submap_pts = torch_to_numpy(gaussian_params["xyz"])
        # rgb = np.zeros_like(submap_pts)
        # rgb[:, 2] = 255
        sh = torch_to_numpy(gaussian_params["features_dc"])
        C0 = 0.28209479177387814
        rgb = sh.squeeze(1) * C0 + 0.5

        point_cloud2 = numpy_to_point_cloud(submap_pts, rgb)
        point_cloud2 = point_cloud2.voxel_down_sample(voxel_size=0.02)

        # o3d.visualization.draw_geometries([point_cloud1, point_cloud2])
        o3d.visualization.draw_geometries([point_cloud2])

        # Create a visualizer
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=False)  # Create a window but do not display it
        # vis.add_geometry(point_cloud)

        # # Render the scene
        # vis.update_geometry()
        # vis.poll_events()
        # vis.update_renderer()

        # # Capture the screen and save the image
        # image = vis.capture_screen()
        # o3d.io.write_image("test_o3d.png", image)

        # # Close the visualizer
        # vis.destroy_window()







if __name__ == "__main__":
    # scene = Scene.from_directory("/home/anonymous/Documents/Research/gaussian-splatting-playground/data/runs/20240529_181701")
    # scene_list = [
    #     "/home/anonymous/Documents/Research/gaussian-splatting-playground/data/runs/room0",
    #     "/home/anonymous/Documents/Research/gaussian-splatting-playground/data/runs/office3",
    #     "/home/anonymous/Documents/Research/gaussian-splatting-playground/data/runs/rgbd_dataset_freiburg3_long_office_household",
    # ]

    scene_list = [
        # "/home/anonymous/Documents/Research/gaussian-splatting-playground/data/runs/office0",
        "/home/anonymous/Documents/Research/gaussian-splatting-playground/data/runs/data",
    ]

    for scene_directory in scene_list:
        scene = Scene.from_directory(scene_directory)
        gaussian_model = scene.assemble_global_map()
        try:
            scene.save_global_map(gaussian_model)
        except:
            breakpoint()