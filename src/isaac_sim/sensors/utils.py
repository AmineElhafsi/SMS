import os
import shutil

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from scipy.interpolate import interp1d

def generate_helical_camera_path(
    center_position: np.ndarray,
    radius: float,
    z_low: float, 
    z_high: float, 
    num_revolutions: int, 
    points_per_revolution: int
):
    pitch = (z_high - z_low) / (num_revolutions)
    s = np.linspace(0, 2 * np.pi * num_revolutions, num_revolutions * points_per_revolution)
    x = radius * np.cos(s) + center_position[0]
    y = radius * np.sin(s) + center_position[1]
    z = s / (2 * np.pi) * pitch + z_low
    return np.vstack((x, y, z)).T.tolist()

def generate_hemispheric_camera_path(
    theta_range, 
    phi_range,
    center=np.array([0., 0., 0.]),
    radius=1.0,
    offset=np.array([0.1, 0., 0.1]), 
    num_elevations=3,
    density=5,
    max_points=80
):
    """
    Generate a hemispheric scan path with a smooth transition between viewpoints.

    Parameters:
        - theta_range: (theta_min, theta_max) in degrees (azimuthal angle).
        - phi_range: (phi_min, phi_max) in degrees (elevation angle).
        - center: (x, y, z) coordinates of the center of the hemisphere.
        - radius: Distance of the camera from the center.
        - offset: (x, y, z) offset to apply to the path.
        - num_elevations: Number of elevation steps.
        - density: Desired number of points per unit arc length.
        - max_points: Maximum number of points in the final path.
    """

    # calculate the arc lengths for theta and phi ranges
    theta_arc_length = np.radians(theta_range[1] - theta_range[0]) * radius

    # determine the number of steps based on density
    num_theta = int(np.ceil(theta_arc_length * density))
    
    # generate theta and phi values
    theta_vals = np.radians(np.linspace(*theta_range, num_theta))
    num_phi = num_elevations
    phi_vals = np.radians(np.linspace(*phi_range, num_phi))
    
    path_points = []

    # establishing shot — directly above, higher elevation
    path_points.append([center[0], center[1], center[2] + radius / 2])

    # generate scanning path
    for phi in phi_vals:
        for theta in theta_vals:
            x = center[0] + radius * np.cos(theta) * np.sin(phi)
            y = center[1] + radius * np.sin(theta) * np.sin(phi)
            z = center[2] + radius * np.cos(phi)
            path_points.append([x, y, z])
    path_points = np.array(path_points)

    # compute distances between consecutive points
    segment_lengths = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    total_points = np.round(segment_lengths * density).astype(int)
    total_points = np.maximum(total_points, 2)

    # interpolating a smooth path
    cumulative_distances = np.insert(np.cumsum(segment_lengths), 0, 0)  # cumulative arc length
    total_samples = np.sum(total_points)

    interp_func = interp1d(cumulative_distances, path_points, axis=0, kind='linear', assume_sorted=True)
    interpolated_distances = np.linspace(0, cumulative_distances[-1], total_samples)
    camera_path = interp_func(interpolated_distances)

    # downsample to max_points
    if max_points is not None and max_points < len(camera_path):
        downsampled_distances = np.linspace(0, cumulative_distances[-1], max_points)
        camera_path = interp_func(downsampled_distances)

    # apply offset
    camera_path += offset

    return camera_path

def camera_path_scan(
    camera,
    camera_positions: np.ndarray,
    camera_pointing: np.ndarray = np.array([0.55, 0.0, 0.]), # default for billiards scene
    output_directory: Optional[str] = None,
    overwrite: bool = False
):
    # create save directories
    # check if output directory already exists
    output_directory = Path(output_directory)
    print("Output directory:", output_directory)
    if output_directory.exists():
        if not overwrite:
            raise ValueError(f"Output directory {output_directory} already exists.")
        else:
            print(f"Output directory {output_directory} already exists. Overwriting.")
            # delete existing directory and create a new one
            shutil.rmtree(output_directory)
            output_directory.mkdir(parents=True)
            os.makedirs(output_directory / "rgb", exist_ok=True)
            os.makedirs(output_directory / "rgb_jpg", exist_ok=True)
            os.makedirs(output_directory / "depth", exist_ok=True)
            # os.makedirs(output_directory / "instance_segmentation", exist_ok=True)
            # os.makedirs(output_directory / "instance_segmentation_id_mapping", exist_ok=True)
    else:
        output_directory.mkdir(parents=True)
        os.makedirs(output_directory / "rgb", exist_ok=True)
        os.makedirs(output_directory / "rgb_jpg", exist_ok=True)
        os.makedirs(output_directory / "depth", exist_ok=True)
        # os.makedirs(output_directory / "instance_segmentation", exist_ok=True)
        # os.makedirs(output_directory / "instance_segmentation_id_mapping", exist_ok=True)

    # generate data
    camera_to_world_transforms = []
    for i, camera_position in enumerate(camera_positions):
        camera.set_world_pose(position=np.array(camera_position))
        camera.point_at(camera_pointing)
        current_frame = camera.get_data(render_steps=30)

        # save data
        color = current_frame["rgba"][:, :, :3]
        depth = current_frame["distance_to_image_plane"]
        # instance_segmentation_dict = current_frame["instance_segmentation"]
        # instance_segmentation, semantics_to_ids = simple_merge_instance_segmentation(instance_segmentation_dict)

        # data formatting
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        # instance_segmentation = instance_segmentation.astype(np.uint16)

        cv2.imwrite(str(output_directory / "rgb" / f"{i:04d}.png"), color)
        cv2.imwrite(str(output_directory / "rgb_jpg" / f"{i:04d}.jpg"), color)
        np.save(str(output_directory / "depth" / f"{i:04d}.npy"), depth)
        # np.save(str(output_directory / "instance_segmentation" / f"{i:04d}.npy"), instance_segmentation)
        # np.save(str(output_directory / "instance_segmentation_id_mapping" / f"{i:04d}_semantics_to_ids.npy"), semantics_to_ids)

        camera_to_world_transforms.append(camera.get_camera_to_world_transform().flatten())
    np.save(output_directory / "camera_to_world_transforms.npy", np.array(camera_to_world_transforms))