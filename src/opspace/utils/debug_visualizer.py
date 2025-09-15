"""Functions associated with the Pybullet Debug Visualizer GUI"""

import time
from typing import Optional

import numpy as np
import numpy.typing as npt
import pybullet
from pybullet_utils.bullet_client import BulletClient


def visualize_points(
    position: npt.ArrayLike,
    color: npt.ArrayLike,
    size: float = 20,
    lifetime: float = 0,
    client: Optional[BulletClient] = None,
) -> int:
    """Adds square points to the GUI to visualize positions in the sim

    Args:
        position (npt.ArrayLike): 3D point(s) in the simulation to visualize. Shape (n, 3)
        color (npt.ArrayLike): RGB values, each in range [0, 1]. Shape (3,) if specifying the same color for all points,
            or (n, 3) to individually specify the colors per-point
        size (float): Size of the points on the GUI, in pixels. Defaults to 20
        lifetime (float, optional): Amount of time to keep the points on the GUI, in seconds.
            Defaults to 0 (keep them on-screen permanently until deleted)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        int: Pybullet object ID for the point / point cloud
    """
    client: pybullet = pybullet if client is None else client
    # Pybullet will crash if you try to visualize one point without packing it into a 2D array
    position = np.atleast_2d(position)
    color = np.atleast_2d(color)
    if position.shape[-1] != 3:
        raise ValueError(
            f"Invalid shape of the point positions. Expected (n, 3), got: {position.shape}"
        )
    if color.shape[-1] != 3:
        raise ValueError(
            f"Invalid shape of the colors. Expected (n, 3), got: {color.shape}"
        )
    n = position.shape[0]
    if color.shape[0] != n:
        if color.shape[0] == 1:
            # Map the same color to all of the points
            color = color * np.ones_like(position)
        else:
            raise ValueError(
                f"Number of colors ({color.shape[0]}) does not match the number of points ({n})."
            )
    return client.addUserDebugPoints(position, color, size, lifetime)


def visualize_frame(
    tmat: np.ndarray,
    length: float = 1,
    width: float = 3,
    lifetime: float = 0,
    client: Optional[BulletClient] = None,
) -> tuple[int, int, int]:
    """Adds RGB XYZ axes to the Pybullet GUI for a speficied transformation/frame/pose

    Args:
        tmat (np.ndarray): Transformation matrix specifying a pose w.r.t world frame, shape (4, 4)
        length (float, optional): Length of the axis lines. Defaults to 1.
        width (float, optional): Width of the axis lines. Defaults to 3. (units unknown, maybe mm?)
        lifetime (float, optional): Amount of time to keep the lines on the GUI, in seconds.
            Defaults to 0 (keep them on-screen permanently until deleted)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        tuple[int, int, int]: Pybullet IDs of the three axis lines added to the GUI
    """
    client: pybullet = pybullet if client is None else client
    x_color = [1, 0, 0]  # R
    y_color = [0, 1, 0]  # G
    z_color = [0, 0, 1]  # B
    origin = tmat[:3, 3]
    x_endpt = origin + tmat[:3, 0] * length
    y_endpt = origin + tmat[:3, 1] * length
    z_endpt = origin + tmat[:3, 2] * length
    x_ax_id = client.addUserDebugLine(origin, x_endpt, x_color, width, lifetime)
    y_ax_id = client.addUserDebugLine(origin, y_endpt, y_color, width, lifetime)
    z_ax_id = client.addUserDebugLine(origin, z_endpt, z_color, width, lifetime)
    return x_ax_id, y_ax_id, z_ax_id


def visualize_path(
    positions: npt.ArrayLike,
    n: Optional[int] = None,
    color: npt.ArrayLike = (1, 0, 0),
    width: float = 3,
    lifetime: float = 0,
    client: Optional[BulletClient] = None,
) -> list[int]:
    """Visualize a sequence of positions on the Pybullet GUI

    Args:
        positions (npt.ArrayLike): Sequence of positions, shape (n, 3)
        n (Optional[int]): Number of lines to plot, if plotting the lines between all positions is not desired.
            Defaults to None (plot all lines between positions)
        color (npt.ArrayLike, optional): RGB color values. Defaults to (1, 0, 0) (red).
        width (float, optional): Width of the line. Defaults to 3 (pixels)
        lifetime (float, optional): Amount of time to keep the lines on the GUI, in seconds.
            Defaults to 0 (keep them on-screen permanently until deleted)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        list[int]: Pybullet IDs of the lines added to the GUI
    """
    client: pybullet = pybullet if client is None else client
    positions = np.atleast_2d(positions)
    n_positions, dim = positions.shape
    assert dim == 3
    # If desired, sample frames evenly across the trajectory to plot a subset
    if n is not None and n < n_positions:
        # This indexing ensures that the first and last frames are plotted
        idx = np.round(np.linspace(0, n_positions - 1, n, endpoint=True)).astype(int)
        positions = positions[idx, :]
    ids = []
    for i in range(positions.shape[0] - 1):
        ids.append(
            client.addUserDebugLine(
                positions[i], positions[i + 1], color, width, lifetime
            )
        )
    return ids


def animate_path(
    positions: npt.ArrayLike,
    duration: float,
    n: Optional[int] = None,
    color: npt.ArrayLike = (1, 1, 1),
    size: float = 20,
    client: Optional[BulletClient] = None,
):
    """Animates a point moving along a sequence of positions

    Args:
        positions (npt.ArrayLike): Path to animate, shape (n, 3)
        duration (float): Desired duration of the animation
        n (Optional[int]): Number of points to use in the animation, if using all of the provided positions will be too
            slow. Defaults to None (animate all points)
        color (npt.ArrayLike): RGB values, each in range [0, 1]. Shape (3,) if specifying the same color for all points,
            or (n, 3) to individually specify the colors per-point
        size (float): Size of the points on the GUI, in pixels. Defaults to 20
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """
    client: pybullet = pybullet if client is None else client
    # Pybullet will crash if you try to visualize one point without packing it into a 2D array
    positions = np.atleast_2d(positions)
    n_positions, dim = positions.shape
    if dim != 3:
        raise ValueError(
            f"Invalid shape of the point positions. Expected (n, 3), got: {positions.shape}"
        )
    color = np.atleast_2d(color)
    if color.shape[-1] != 3:
        raise ValueError(
            f"Invalid shape of the colors. Expected (n, 3), got: {color.shape}"
        )
    if color.shape[0] != n_positions:
        if color.shape[0] == 1:
            # Map the same color to all of the points
            color = color * np.ones_like(positions)
        else:
            raise ValueError(
                f"Number of colors ({color.shape[0]}) does not match the number of points ({n_positions})."
            )
    # Downsample the points if desired
    if n is not None and n < n_positions:
        # This indexing ensures that the first and last frames are plotted
        idx = np.round(np.linspace(0, n_positions - 1, n, endpoint=True)).astype(int)
        positions = positions[idx, :]
        color = color[idx, :]
        n_positions = n
    uid = None
    for i in range(n_positions):
        start_time = time.time()
        if uid is None:
            uid = client.addUserDebugPoints([positions[i]], [color[i]], size, 0)
        else:
            uid = client.addUserDebugPoints(
                [positions[i]], [color[i]], size, 0, replaceItemUniqueId=uid
            )
        client.stepSimulation()
        elapsed_time = time.time() - start_time
        time.sleep(max(0, duration / n_positions - elapsed_time))
