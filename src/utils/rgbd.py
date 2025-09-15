import cv2
import numpy as np

def depth_to_pointcloud(depth: np.ndarray, camera_intrinsics: np.ndarray, T_c2w: np.ndarray) -> np.ndarray:
    """
    Convert depth map to pointcloud in world coordinates.

    Args:
        depth (np.ndarray): Depth map of shape (H, W).
        camera_intrinsics (np.ndarray): Camera intrinsics of shape (3, 3).
        T_c2w (np.ndarray): Camera-to-world transformation matrix of shape (4, 4).
    Returns:
        np.ndarray: Pointcloud in world coordinates of shape (N, 3).
    """
    # compute get pixel coordinates
    h, w = depth.shape
    u = np.arange(0, w)
    v = np.arange(0, h)
    u_grid, v_grid = np.meshgrid(u, v)
    u_grid, v_grid = u_grid.flatten(), v_grid.flatten()

    # convert to camera coordinates
    z_cam = depth.flatten()
    x_cam = (u_grid - camera_intrinsics[0, 2]) * z_cam / camera_intrinsics[0, 0] # (u_grid - cx) * z_cam / fx
    y_cam = (v_grid - camera_intrinsics[1, 2]) * z_cam / camera_intrinsics[1, 1] # (v_grid - cy) * z_cam / fy

    # convert to world coordinates
    p_cam = np.vstack((x_cam, y_cam, z_cam, np.ones_like(x_cam)))
    p_world = (T_c2w @ p_cam).T
    pointcloud = p_world[:, :3]

    return pointcloud

def get_normals(depth: np.ndarray, camera_intrinsics: np.ndarray, T_c2w: np.ndarray) -> np.ndarray:
    """
    Compute normals from pointcloud using cross product of gradients in x and y direction.

    Args:
        depth (np.ndarray): Depth map of shape (H, W).
        camera_intrinsics (np.ndarray): Camera intrinsics of shape (3, 3).
        T_c2w (np.ndarray): Camera-to-world transformation matrix of shape (4, 4).
    Returns:
        np.ndarray: Normal map of shape (H, W, 3).
    """
    # construct point cloud in world coordinates
    pointcloud = depth_to_pointcloud(depth, camera_intrinsics, T_c2w).reshape(*depth.shape, 3)

    # compute gradients using Sobel operator
    dx = cv2.Sobel(pointcloud, cv2.CV_64F, 0, 1, ksize=3)
    dy = cv2.Sobel(pointcloud, cv2.CV_64F, 1, 0, ksize=3)
    
    # compute normals using cross product
    normal_map = np.cross(dx, dy)
    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map = normal_map / (norm + (norm < 1e-8) * 1e-8)  # Normalize and avoid division by zero
    
    return normal_map