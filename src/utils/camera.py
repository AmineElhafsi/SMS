import math

import numpy as np
import torch


def get_camera_intrinsics_matrix(config: dict) -> np.ndarray:
    """
    Returns the camera intrinsics matrix from a configuration dictionary.
    Args:
        config: A dictionary containing the camera intrinsics parameters.
    Returns:
        K: The camera intrinsics matrix.
    """
    # if config["mode"] == "dataset":
    fx, fy = config["dataset_config"]["fx"], config["dataset_config"]["fy"]
    cx, cy = config["dataset_config"]["cx"], config["dataset_config"]["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
    return K


def world_points_to_camera_coordinates(
    p_world: torch.Tensor, T_w2c: torch.Tensor
) -> torch.Tensor:
    """
    Converts world points to camera coordinates.
    Args:
        p_world: World points.
        K: Camera intrinsics matrix.
        T_w2c: World-to-camera transform.
    Returns:
        p_camera: Camera coordinates.
    """
    p_world_homogeneous = torch.cat([p_world, torch.ones(p_world.shape[0], 1, device=p_world.device)], dim=1)
    p_camera = (T_w2c @ p_world_homogeneous.T).T[:, :3]
    return p_camera


def world_points_to_pixel_coordinates(
    p_world: torch.Tensor, K: torch.Tensor, T_w2c: torch.Tensor
) -> torch.Tensor:
    """
    Converts world points to pixel coordinates.
    Args:
        p_w: World points.
        K: Camera intrinsics matrix.
        T_w2c: World-to-camera transform.
    Returns:
        pixel_coordinates: Pixel coordinates.
    """
    # device = p_world.device

    # p_world_homogeneous = torch.cat([p_world, torch.ones(p_world.shape[0], 1, device=device)], dim=1) # world
    # p_camera_homogeneous = (T_w2c @ p_world_homogeneous.T).T # camera
    p_camera = world_points_to_camera_coordinates(p_world, T_w2c) # camera
    p_pixel_homogeneous = (K @ p_camera.T).T # pixel
    p_pixel = (p_pixel_homogeneous[:, :2] / p_pixel_homogeneous[:, 2:]).type(torch.int)    

    return p_pixel


def estimate_visible_points(
    p_world: torch.Tensor, depth: torch.Tensor, K: torch.Tensor, T_w2c: torch.Tensor, 
) -> torch.Tensor:
    """
    Estimates the visible (unoccluded) points in the image plane. The depth image is used to determine the visibility.

    Args:
        p_world: World points.
        depth: Depth image.
        K: Camera intrinsics matrix.
        T_w2c: World-to-camera transform.
        height: Image height.
        width: Image width.
    Returns:
        visible_mask: Mask indicating the visibility of the points.
    """
    device = p_world.device
    height, width = depth.shape

    p_pixel = world_points_to_pixel_coordinates(p_world, K, T_w2c)
    visible_mask = (p_pixel[:, 0] >= 0) & (p_pixel[:, 0] < width) & (p_pixel[:, 1] >= 0) & (p_pixel[:, 1] < height)

    # remove occluded points from mask
    p_camera = world_points_to_camera_coordinates(p_world, T_w2c)

    p_pixel = p_pixel[visible_mask]
    p_depth = depth[p_pixel[:, 1], p_pixel[:, 0]]
    occluded_mask = p_camera[visible_mask, 2] > (p_depth+0.1)
    visible_mask[visible_mask.nonzero()[occluded_mask]] = False
    
    return visible_mask


def depth_image_to_pointcloud(
    depth: torch.Tensor, K: torch.Tensor, T_c2w: torch.Tensor
) -> torch.Tensor:
    """
    Converts a depth image to a point cloud.
    Args:
        depth: Depth image.
        K: Camera intrinsics matrix.
    Returns:
        pointcloud: Point cloud.
    """
    device = depth.device
    height, width = depth.shape

    # create a grid of pixel coordinates
    u, v = torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy")
    u, v = u.to(depth.device), v.to(depth.device)

    # pixel to (homogeneous) camera coordinates
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = (v - K[1, 2]) * depth / K[1, 1]
    z = depth
    p_camera = torch.stack([x, y, z, torch.ones_like(z)], axis=-1)
    p_camera = p_camera.reshape(-1, 4)

    # camera to world coordinates
    p_world = (T_c2w @ p_camera.T).T[:, :3]
    return p_world


def rotation_to_euler(R: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotation matrix to Euler angles.
    Args:
        R: A rotation matrix.
    Returns:
        Euler angles corresponding to the rotation matrix.
    """
    sy = torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0

    return torch.tensor([x, y, z]) * (180 / math.pi)


def exceeds_motion_thresholds(
    Tc2w_current: torch.Tensor,
    Tc2w_reference: torch.Tensor,
    translation_threshold: float = 0.5,
    rotation_threshold: float = 50.0,
) -> bool:
    """
    Checks if the current pose exceeds the rotation and translation thresholds from a 
    reference pose. 
    Args:
        Tc2w_current (torch.Tensor): Current camera-to-world transform.
        Tc2w_reference (torch.Tensor): Reference camera-to-world transform.
        translation_threshold (float): Translation threshold in meters.
        rotation_threshold (float): Rotation threshold in degrees.
    Returns:
        exceeds_thresholds: A boolean indicator of whether the pose difference exceeds the specified 
        translation or rotation thresholds.
    """
    Tw2c_reference = torch.linalg.inv(Tc2w_reference).float()
    T_diff = Tw2c_reference @ Tc2w_current # T_diff transforms current camera to reference camera

    translated_distance = torch.norm(T_diff[:3, 3])
    rotated_distance = torch.abs(rotation_to_euler(T_diff[:3, :3]))
    print("Translated distance: ", translated_distance)
    print("Rotated distance: ", rotated_distance)

    translation_exceeded = (translated_distance > translation_threshold)
    rotation_exceeded = torch.any(rotated_distance > rotation_threshold)
    result = (translation_exceeded or rotation_exceeded).item()
    return result
