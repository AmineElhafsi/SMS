from typing import Optional

import numpy as np
import pytorch3d.ops
import torch

from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, square

from src.utils.camera import depth_image_to_pointcloud
from src.utils.utils import numpy_to_torch, torch_to_numpy

@torch.no_grad()
def refine_segmentation(segmentation: np.ndarray, depth: np.ndarray, K_camera: np.ndarray, T_c2w: np.ndarray, edge_dilation: int = 7, k_knn: int = 5) -> np.ndarray:

    # get point cloud
    point_cloud = depth_image_to_pointcloud(
        numpy_to_torch(depth, device="cuda"), 
        numpy_to_torch(K_camera, device="cuda"), 
        numpy_to_torch(T_c2w, device="cuda")
    )

    # 1) correct mask edges
    # find edges between segments
    edges = find_boundaries(segmentation, connectivity=segmentation.ndim, mode='thick')
    if edge_dilation > 0:
        edges = dilation(edges, square(7))

    # determine pixel classification via knn on point cloud
    depth_nan_mask = np.isnan(depth.reshape(-1))
    edge_mask = (edges.reshape(-1) == True) & (~depth_nan_mask)
    interior_mask = ~edge_mask & (~depth_nan_mask)
    knn = pytorch3d.ops.knn_points(
        point_cloud[edge_mask].unsqueeze(0), 
        point_cloud[interior_mask].unsqueeze(0), 
        K=k_knn
    )
    knn_idx = knn.idx.squeeze(0).cpu()
    nearest_k_idx = interior_mask.nonzero()[0][knn_idx]

    pixel_classes = torch.tensor(segmentation).to("cuda").reshape(-1)
    votes = pixel_classes[nearest_k_idx].mode(dim=1).values.squeeze(0)

    pixel_classes[edge_mask] = votes
    refined_segmentation = torch_to_numpy(pixel_classes.reshape(*segmentation.shape))

    return refined_segmentation
    
    

