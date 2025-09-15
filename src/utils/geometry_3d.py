from typing import List, Optional, Union

import math

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from pytorch3d.transforms import quaternion_to_matrix
from scipy.spatial import Delaunay


def truncate_point_cloud(
    points: np.ndarray, 
    x_bounds: Optional[Union[np.ndarray, List]] = np.array([-np.inf, np.inf]),
    y_bounds: Optional[Union[np.ndarray, List]] = np.array([-np.inf, np.inf]), 
    z_bounds: Optional[Union[np.ndarray, List]] = np.array([-np.inf, np.inf]), 
    #ground_level = 0.01
) -> np.ndarray:
    """
    Filter point cloud to remove points that are outside of specified limits

    Args:
        points (np.ndarray): Nx3 array of points
        x_bounds (np.ndarray or List): x-axis bounds
        y_bounds (np.ndarray or List): y-axis bounds
        z_bounds (np.ndarray or List): z-axis bounds

    Returns:
        filtered_points (np.ndarray): filtered point cloud
    """
    # remove outliers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # remove points that are too far from the origin (measuring within the xy plane)
    # distance_from_origin = np.linalg.norm(np.asarray(pcd.points)[:, :2], axis=1)
    # out_of_bounds = distance_from_origin > xy_bounds
    out_of_bounds = np.zeros(len(points), dtype=bool)    
    out_of_bounds = out_of_bounds | (points[:, 0] < x_bounds[0]) | (points[:, 0] > x_bounds[1])
    out_of_bounds = out_of_bounds | (points[:, 1] < y_bounds[0]) | (points[:, 1] > y_bounds[1])
    out_of_bounds = out_of_bounds | (points[:, 2] < z_bounds[0]) | (points[:, 2] > z_bounds[1])
    pcd = pcd.select_by_index(np.where(out_of_bounds == False)[0])
    
    # remove outliers
    # pcd, ind = pcd.remove_radius_outlier(nb_points=15, radius=0.05)
    filtered_points = np.asarray(pcd.points)

    return filtered_points


def trim_point_cloud_edges(points, edge_proportion, eps=None, axes="z"):
    """
    Trim point cloud to remove stray points at the edges of the shape.

    Args:
        points (np.ndarray): Nx3 array of points
        edge_proportion (float): proportion of (total) point cloud points to consider at the edges of the shape
        eps (float): threshold for identifying stray points
        axes (str): axes along which to trim the point cloud
    
    Returns:
        trimmed_points (np.ndarray): trimmed point cloud
    """
    trimmed_points = points.copy()
    edge_points = int(edge_proportion * trimmed_points.shape[0])

    # make sure axes is a string that only contains 'x', 'y', and/or 'z'
    assert all([axis in "xyz" for axis in axes])

    axes_indices = []
    if "x" in axes:
        axes_indices.append(0)
    if "y" in axes:
        axes_indices.append(1)
    if "z" in axes:
        axes_indices.append(2)


    # trim point cloud in each coordinate to remove stray points at the edges 
    # of the shape
    for i in axes_indices:
        edge_points = int(edge_proportion * trimmed_points.shape[0])

        # sort points wrt coordinates along current axis i
        sorted_indices = np.argsort(trimmed_points[:, i])

        points_sorted = trimmed_points[sorted_indices]
        dp = np.diff(points_sorted[:, i])
        if eps is None:
            eps = 3.5 * np.std(dp) + np.mean(dp)
    
        dp_leading = np.diff(points_sorted[:edge_points, i])
        dp_trailing = -np.diff(points_sorted[-edge_points:, i][::-1])

        start_truncation_index = 0
        end_truncation_index = len(points_sorted)

        if np.any(dp_leading > eps):
            start_truncation_index = np.argmax(dp_leading > eps) + 1
        if np.any(dp_trailing > eps):
            end_truncation_index = np.argmin(dp_trailing > eps) + len(points_sorted) - edge_points + 1
        
        trimmed_points = trimmed_points[sorted_indices[start_truncation_index:end_truncation_index]]

    return trimmed_points


def compute_local_feature_size(points, k=10):
    """
    Compute the local feature size (LFS) for each point in the point cloud.
    
    Args:
        points (np.ndarray): Nx3 array of points
        k (int): Number of nearest neighbors to consider
    
    Returns:
        lfs (np.ndarray): Local feature size for each point
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    lfs = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        [_, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], k)
        neighbors = points[idx[1:], :]  # Exclude the point itself
        distances = np.linalg.norm(neighbors - points[i], axis=1)
        lfs[i] = np.mean(distances)
    
    return lfs


@torch.no_grad()
def point_cloud_to_triangle_mesh(points, max_edge_length=None, adaptive_edge_pruning=True, orient_normals=True, mesh_points_only=True):
    """
    Convert a point cloud to a triangle mesh using Delaunay tetrahedralization. There are four main steps:
    1. Compute the tetrahedralization of the point cloud.
    2. Remove tetrahedra with edges longer than the maximum edge length.
    3. Extract the surface triangles from the tetrahedra.
    4. (Optional) Orient the normals of the triangle mesh faces.

    Args:
        points (np.ndarray): Nx3 array of points
        max_edge_length (float): maximum edge length for tetrahedra
        orient_normals (bool): whether to orient the normals of the triangle mesh faces
    
    Returns:
        mesh_vertices (torch.Tensor): Nx3 array of mesh vertices
        surface_triangles (torch.Tensor): Mx3 array of triangle mesh faces
    """
    
    # compute tetrahedralization
    delaunay = Delaunay(points)
    tetrahedra = torch.tensor(delaunay.simplices, dtype=torch.long, device="cuda") # vertex indices (N_t, 4)

    # compute edge lengths
    points = torch.tensor(points, dtype=torch.float, device="cuda") # points (N_p, 3)
    tetra_vertices = points[tetrahedra] # tetrahedra vertices (N_t, 4, 3)

    # get all 6 edges in each tetrahedron
    edge_indices = torch.combinations(torch.arange(4), r=2)
    edge_lengths = torch.norm(tetra_vertices[:, edge_indices[:, 0]] - tetra_vertices[:, edge_indices[:, 1]], dim=2) # edge lengths (N_t, 6)

    ####
    if adaptive_edge_pruning:
        max_edge_length_factor = 2.25
        # Compute local feature size (LFS)
        lfs = compute_local_feature_size(points.cpu().numpy())
        lfs = torch.tensor(lfs, dtype=torch.float, device="cuda")
        # Adapt edge length threshold based on LFS
        max_edge_lengths = lfs[tetrahedra].max(dim=1)[0] * max_edge_length_factor
        valid_tetrahedra_mask = torch.all(edge_lengths <= max_edge_lengths.unsqueeze(1), dim=1)
    else:
        valid_tetrahedra_mask = torch.all(edge_lengths <= max_edge_length, dim=1) # valid tetrahedra mask (N_t,)
    valid_tetrahedra = tetrahedra[valid_tetrahedra_mask] # vertex indices (N_vt, 4)
    ####

    # mask out tetrahedra with edges longer than max_edge_length
    # valid_tetrahedra_mask = torch.all(edge_lengths <= max_edge_length, dim=1) # valid tetrahedra mask (N_t,)
    # valid_tetrahedra = tetrahedra[valid_tetrahedra_mask] # vertex indices (N_vt, 4)

    # extract surface faces
    face_indices = torch.tensor([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], dtype=torch.long)
    triangles = valid_tetrahedra[:, face_indices] # vertex indices (N_vt, 4, 3)
    triangles = triangles.view(-1, 3) # vertex indices (4*N_vt, 3)
    triangles, _ = torch.sort(triangles, dim=1)

    unique_faces, inverse_indices, counts = torch.unique(triangles, return_inverse=True, return_counts=True, dim=0)
    surface_triangle_mask = (counts[inverse_indices] == 1)
    surface_triangles = triangles[surface_triangle_mask]

    # adjust mesh normals by flipping faces
    if orient_normals:
        # get surface tetrahedra
        valid_tetrahedra_indices = torch.arange(valid_tetrahedra.shape[0], device="cuda").repeat_interleave(4) # vertex indices (4*N_vt, 4)
        surface_tetra_indices = valid_tetrahedra_indices[surface_triangle_mask]
        surface_tetrahedra = valid_tetrahedra[surface_tetra_indices]

        # compute face normals
        v1 = points[surface_triangles[:, 1]] - points[surface_triangles[:, 0]]
        v2 = points[surface_triangles[:, 2]] - points[surface_triangles[:, 0]]
        triangle_normals = F.normalize(torch.linalg.cross(v1, v2), dim=1)

        # compute face and tetrahedra centroids
        triangle_centroids = points[surface_triangles].mean(dim=1)
        tetra_centroids = points[surface_tetrahedra].mean(dim=1)
        outward_direction = triangle_centroids - tetra_centroids

        # flip faces with normals pointing inwards
        dot_products = torch.sum(triangle_normals * outward_direction, dim=1)
        flip_mask = (dot_products < 0)
        surface_triangles[flip_mask] = surface_triangles[flip_mask][:, [0, 2, 1]]
    
    if mesh_points_only:
        mesh_vertex_indices = torch.unique(surface_triangles)

        index_mapping = torch.zeros(points.shape[0], dtype=torch.long, device="cuda")
        index_mapping[mesh_vertex_indices] = torch.arange(mesh_vertex_indices.shape[0], device=points.device)
        surface_triangles  = index_mapping[surface_triangles]

        mesh_vertices = points[mesh_vertex_indices]

    mesh_vertices = mesh_vertices.cpu().numpy()
    surface_triangles = surface_triangles.cpu().numpy()

    return mesh_vertices, surface_triangles

def point_cloud_to_mesh(point_cloud, smooth_mesh):
    """
    Convert a point cloud to a triangle mesh. The point cloud is first filtered and trimmed,
    then converted to a triangle mesh using Delaunay tetrahedralization. The mesh can
    be optionally smoothed using Taubin smoothing.

    Args:
        point_cloud (np.ndarray): Nx3 array of points
        smooth_mesh (bool): whether to smooth the mesh
    
    Returns:
        mesh_vertices (np.ndarray): Nx3 array of mesh vertices
        surface_triangles (np.ndarray): Mx3 array of triangle mesh faces
    """

    # preprocess point cloud
    filtered_points = filter_point_cloud(point_cloud, xy_bounds=1, ground_level=0.01)
    trimmed_points = trim_point_cloud_edges(filtered_points, edge_proportion=0.001)

    # convert point cloud to triangle mesh
    mesh_vertices, surface_triangles = point_cloud_to_triangle_mesh(trimmed_points, max_edge_length=0.0375)

    if smooth_mesh:
        # create o3d mesh object
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(surface_triangles)
        smoothed_mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
        mesh_vertices = np.asarray(smoothed_mesh.vertices)
        surface_triangles = np.asarray(smoothed_mesh.triangles)
    
    return mesh_vertices, surface_triangles


def densify_point_cloud(
    gaussian_centers: torch.Tensor, # gaussian_model.get_xyz()
    gaussian_rotations: torch.Tensor, # gaussian_model.get_rotation()
    gaussian_scales: torch.Tensor, # gaussian_model.get_scaling()
    density: float,
    control_points_extent: float = 3.0,
):
    """
    Sample control points along the two largest principal axes of each gaussian.

    Args:
        gaussian_rotations (torch.Tensor): Rotations of the gaussians as quaternions (real part first).
        gaussian_scalings (torch.Tensor): Scaling coefficients of the gaussians.
        gaussian_centers (torch.Tensor): Centers of the gaussians.
        control_points_extent (float, optional): Extent of control points along the principal axes. Defaults to 3.0.
    
    Returns:
        torch.Tensor: Control points
    """
    # get rotations, scaling, centers of gaussians
    P = gaussian_centers
    R = quaternion_to_matrix(gaussian_rotations)
    S = gaussian_scales
    n_gaussians = R.shape[0]
    device = R.device

    # compute areas of ellipses defined by the two largest principal axes
    principal_scales, indices = torch.topk(S, k=2, dim=1, largest=True, sorted=True)
    areas = np.pi * (control_points_extent * principal_scales).prod(dim=1)

    # compute number of points to sample for each gaussian
    n_points = (density * areas).ceil().int()

    # sample max points in (0, 1] X [0, 2*pi]
    sampled_points = torch.rand((n_points.max(), 2), device=device)
    r_sample = (1 - sampled_points[:, 0])
    theta_sample = 2 * np.pi * sampled_points[:, 1]

    # compute x_body and y_body for each Gaussian
    a = control_points_extent * principal_scales[:, 0].unsqueeze(1)
    b = control_points_extent * principal_scales[:, 1].unsqueeze(1)
    x_body = r_sample * a * torch.cos(theta_sample) # (n_gaussians, n_points.max())
    y_body = r_sample * b * torch.sin(theta_sample)

    # create a random mask for each Gaussian
    rand_indices = torch.rand((n_gaussians, n_points.max()), device=device).argsort(dim=1)
    mask = torch.arange(n_points.max(), device=device).expand(n_gaussians, n_points.max()) < n_points.unsqueeze(1)
    mask = mask.gather(1, rand_indices)

    # apply mask to x_body and y_body
    x_body = x_body[mask]
    y_body = y_body[mask]

    # create control points in the body frame
    # control_points_body = torch.stack([x_body, y_body, torch.zeros_like(x_body)], dim=1)
    control_points_body = torch.stack([x_body, y_body], dim=1)

    # extract the principal axes corresponding to the largest scales
    principal_axes = torch.gather(R, 2, indices.unsqueeze(1).expand(-1, 3, -1))

    # transform control points to the world frame
    # repeat the principal axes and centers according to the number of points for each Gaussian
    principal_axes_repeated = principal_axes.repeat_interleave(n_points, dim=0)
    P_repeated = P.repeat_interleave(n_points, dim=0)

    # Transform the control points to the world frame
    control_points = (principal_axes_repeated @ control_points_body.unsqueeze(-1)).squeeze() + P_repeated

    return control_points