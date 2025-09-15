import os
import random

import numpy as np
import open3d
import torch


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


def setup_seed(seed: int) -> None:
    """
    Sets the seed for reproducibility.
    Args:
        seed (int): Seed value for torch, numpy, and random.
    """
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array.
    Args:
        tensor (torch.Tensor): Input tensor.
    Returns:
        np.ndarray: NumPy array.
    """
    return tensor.detach().cpu().numpy()


def numpy_to_point_cloud(points: np.ndarray, rgb=None) -> open3d.geometry.PointCloud:
    """
    Converts NumPy array to o3d point cloud.

    Args:
        points (ndarray): Point cloud as an array.
    Returns:
        (PointCloud): PointCloud in o3d format.
    """
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(points)
    if rgb is not None:
        cloud.colors = open3d.utility.Vector3dVector(rgb)
    return cloud


def numpy_to_torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Converts a NumPy array to a PyTorch tensor.
    Args:
        array (np.ndarray): Input array.
        device (str): Device to store the tensor.
    Returns:
        torch.Tensor: PyTorch tensor.
    """
    return torch.from_numpy(array).float().to(device)

if __name__ == "__main__":
    setup_seed(42)
    print("Seed is set.")