from itertools import cycle

import faiss
import faiss.contrib.torch_utils # added to avoid errors with gpu vs. cpu arrays
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from src.splatting.gaussian_model import GaussianModel
from src.utils.metrics import isotropic_loss, l1_loss, ssim
from src.splatting.parameters import OptimizationParams
from src.utils.rendering import render_gaussian_model
from src.utils.utils import numpy_to_point_cloud, torch_to_numpy


def batch_search_faiss(indexer, query_points, k):
    """
    Perform a batch search on a IndexIVFFlat indexer to circumvent the search size limit of 65535.

    Args:
        indexer: The FAISS indexer object.
        query_points: A tensor of query points.
        k (int): The number of nearest neighbors to find.

    Returns:
        distances (torch.Tensor): The distances of the nearest neighbors.
        ids (torch.Tensor): The indices of the nearest neighbors.
    """
    split_pos = torch.split(query_points, 65535, dim=0)
    distances_list, ids_list = [], []

    for split_p in split_pos:
        distance, id = indexer.search(split_p.float(), k)
        distances_list.append(distance.clone())
        ids_list.append(id.clone())
    distances = torch.cat(distances_list, dim=0)
    ids = torch.cat(ids_list, dim=0)

    return distances, ids


def merge_submaps(submaps: list, radius: float = 0.0001, max_points: int = 1000000, device: str = "cuda") -> o3d.geometry.PointCloud:
    """
    Merges submaps into a single global map.

    Args:
        submaps (list): List of submaps.
        radius (float): Nearest neighbor distance for adding points.
        device (str): Device to store the tensors.
    """
    # prepare vector database for efficient neighbor search
    if device == "cpu":
        pts_index = faiss.IndexFlatL2(3)
    elif device == "cuda":
        pts_index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            0,
            faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, 500, faiss.METRIC_L2)
        )
        pts_index.nprobe = 5
    else:
        raise ValueError(f"Invalid device: {device}")
    
    # create a merge point cloud
    merged_pts = []
    print("Creating global map from submaps...")
    for submap in tqdm(submaps):
        gaussian_params = submap["gaussian_params"]
        submap_pts = gaussian_params["xyz"].to(device).float()
        pts_index.train(submap_pts)
        distances, _ = batch_search_faiss(pts_index, submap_pts, 8)
        num_neighbors = (distances < radius).sum(axis=1).int()
        ids_to_include = torch.where(num_neighbors == 0)[0]
        pts_index.add(submap_pts[ids_to_include])
        merged_pts.append(submap_pts[ids_to_include])
    
    global_points = torch_to_numpy(torch.vstack(merged_pts))
    point_cloud = numpy_to_point_cloud(global_points, np.zeros_like(global_points))

    # ensure the number of points is within the limit
    if len(point_cloud.points) > max_points:
        print(f"Global map point cloud has {len(point_cloud.points)} points")
        # point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01) # 0.02 original voxel size
        print(f"Global map point cloud downsampled to {len(point_cloud.points)} points")
    filtered_point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=3.0)
    print(f"Global map point cloud filtered to {len(filtered_point_cloud.points)} points")
    del pts_index
    return filtered_point_cloud
    

def refine_global_map(point_cloud: o3d.geometry.PointCloud, training_keyframes: list, opacity_threshold: float, refinement_iterations: int) -> GaussianModel:
    opt = OptimizationParams()

    gaussian_model = GaussianModel(3)
    gaussian_model.training_setup(opt)
    gaussian_model.add_points(point_cloud)

    training_keyframes = cycle(training_keyframes)
    print("Refining global map...")
    for i in tqdm(range(refinement_iterations)):
        keyframe = next(training_keyframes)

         # render model
        render_dict = render_gaussian_model(gaussian_model, keyframe["render_settings"])
        rendered_image, rendered_depth = render_dict["color"], render_dict["depth"]
        gt_image = keyframe["color_torch"]
        gt_depth = keyframe["depth_torch"] # TODO: converted keyframe elements to tensor

        # mask out invalid depth values
        mask = (gt_depth > 0) & (~torch.isnan(rendered_depth)).squeeze(0)

        # compute depth loss
        depth_loss = l1_loss(rendered_depth[:, mask], gt_depth[mask])

        # compute color loss
        weight = opt.lambda_dssim
        pixelwise_color_loss = l1_loss(rendered_image[:, mask], gt_image[:, mask])
        ssim_loss = (1.0 - ssim(rendered_image, gt_image)) # TODO: check why mask isn't used here
        color_loss = (1.0 - weight) * pixelwise_color_loss + weight * ssim_loss

        # compute isotropic regularization loss
        isotropic_regularization_loss = isotropic_loss(gaussian_model.get_scaling())

        # compute total loss (assume uniform weighting across all terms)
        total_loss = color_loss + depth_loss + isotropic_regularization_loss

        # backpropagate
        total_loss.backward()

        with torch.no_grad():
            if i % 500 == 0:
                remove_mask = (gaussian_model.get_opacity() < opacity_threshold).squeeze()
                gaussian_model.remove_points(remove_mask)

            # Optimizer step
            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad(set_to_none=True)

    return gaussian_model



