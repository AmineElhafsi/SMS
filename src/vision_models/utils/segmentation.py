import cv2
import matplotlib.pyplot as plt
import numpy as np

from typing import List

def get_background_mask(
    foreground_masks: np.ndarray,
    depth: np.ndarray, 
    T_c2w: np.ndarray, 
    camera_intrinsics: np.ndarray, 
    distance_threshold: float = 3
) -> np.ndarray:
    """
    Returns a background mask which consists of pixels that are not part of the foreground masks
    and further than a distance threshold from the camera.

    Args:
        foreground_masks: List of foreground masks.
        depth: Depth image.
        T_c2w: Camera-to-world transformation matrix.
        camera_intrinsics: Camera intrinsics matrix.
        ground_height: Height of the ground.
    Returns:
        np.ndarray: Ground mask.
    """
    # compute get pixel coordinates
    h, w = depth.shape
    u = np.arange(0, w)
    v = np.arange(0, h)
    u_grid, v_grid = np.meshgrid(u, v)
    u_grid, v_grid = u_grid.flatten(), v_grid.flatten()

    # convert to camera coordinates
    z_cam = depth.flatten()
    x_cam = (u_grid - camera_intrinsics[0, 2]) * z_cam / camera_intrinsics[0, 0] # (u_grid - cx) * depth.flatten() / fx
    y_cam = (v_grid - camera_intrinsics[1, 2]) * z_cam / camera_intrinsics[1, 1] # (v_grid - cy) * depth.flatten() / fy

    p_cam = np.vstack((x_cam, y_cam, z_cam))
    d_cam = np.linalg.norm(p_cam, axis=0)

    distance_mask = (d_cam > distance_threshold).reshape(h, w)
    background_mask = ~np.any(foreground_masks,axis=0) & distance_mask

    return background_mask

def get_ground_mask(
    depth: np.ndarray, 
    T_c2w: np.ndarray, 
    camera_intrinsics: np.ndarray, 
    ground_height: int = 0.005 # 0.015
) -> np.ndarray:
    """
    Segment ground from depth image assuming the ground is flat and at a fixed height.

    Args:
        depth: Depth image.
        T_c2w: Camera-to-world transformation matrix.
        camera_intrinsics: Camera intrinsics matrix.
        ground_height: Height of the ground.
    Returns:
        np.ndarray: Ground mask.
    """
    # compute get pixel coordinates
    h, w = depth.shape
    u = np.arange(0, w)
    v = np.arange(0, h)
    u_grid, v_grid = np.meshgrid(u, v)
    u_grid, v_grid = u_grid.flatten(), v_grid.flatten()

    # convert to camera coordinates
    z_cam = depth.flatten()
    x_cam = (u_grid - camera_intrinsics[0, 2]) * z_cam / camera_intrinsics[0, 0] # (u_grid - cx) * depth.flatten() / fx
    y_cam = (v_grid - camera_intrinsics[1, 2]) * z_cam / camera_intrinsics[1, 1] # (v_grid - cy) * depth.flatten() / fy

    p_cam = np.vstack((x_cam, y_cam, z_cam, np.ones_like(x_cam)))
    p_world = (T_c2w @ p_cam).T

    ground_mask = (p_world[:, 2] < ground_height).reshape(h, w)

    return ground_mask

def fill_holes(array: np.ndarray, fill_mask: np.ndarray) -> np.ndarray:
    """
    Fill holes of array of entries with designated fill value.

    Args:
        array (np.ndarray): The input array containing integer values.
        fill_mask (np.ndarray): The mask where the array should be filled.

    Returns:
        np.ndarray: The filled array.
    """
    inpainted_array = cv2.inpaint(
        array.copy().astype(np.uint8), 
        fill_mask.astype(np.uint8), 
        inpaintRadius=1, 
        flags=cv2.INPAINT_NS
    )
    array[fill_mask] = inpainted_array[fill_mask]

    return array

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_mask(mask, ax, color=None, random_color=False, borders=True):
    # set random seed
    if color is None:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   