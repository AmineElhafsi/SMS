import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F

from typing import List, Tuple, Union, Optional

from PIL import Image
from torchvision.utils import draw_bounding_boxes


def eliminate_bbox_overlap(boxes: np.ndarray, overlap_ratio_threshold: int = 0.8) -> np.ndarray:
    """
    Checks bounding boxes for overlaps. If two bounding boxes overlap, eliminate the smaller one.

    Parameters:
    boxes (np.array): Tensor of bounding boxes with shape (N, 4)

    Returns:
    np.array: Adjusted bounding boxes (M, 4) where M <= N
    """
    
    adjusted_boxes = boxes.copy()
    n_boxes = boxes.shape[0]

    for i in range(n_boxes):
        for j in range(i+1, n_boxes):
            # compute overlap area
            area_overlap = calculate_bbox_overlap_area(adjusted_boxes[i], adjusted_boxes[j])

            # compute proportion of smaller box covered by overlap
            box1_area = (adjusted_boxes[i][2] - adjusted_boxes[i][0]) * (adjusted_boxes[i][3] - adjusted_boxes[i][1])
            box2_area = (adjusted_boxes[j][2] - adjusted_boxes[j][0]) * (adjusted_boxes[j][3] - adjusted_boxes[j][1])
            smaller_box_area = min(box1_area, box2_area)
            overlap_ratio = area_overlap / smaller_box_area

            # if overlap is significant wrt smaller box, eliminate smaller box
            if area_overlap > 0 and overlap_ratio > overlap_ratio_threshold:
                if box1_area > box2_area:
                    adjusted_boxes[j] = np.zeros(4)
                else:
                    adjusted_boxes[i] = np.zeros(4)

    eliminated_box_indices = np.where(np.all(adjusted_boxes == 0, axis=1))[0]
    adjusted_boxes = adjusted_boxes[~np.all(adjusted_boxes == 0, axis=1)]

    return adjusted_boxes, eliminated_box_indices

def adjust_bboxes(boxes: np.ndarray, overlap_ratio_threshold: int = 0.2) -> np.ndarray:
    """
    Adjust bounding boxes to remove overlaps by modifying their vertical (ymin or ymax) bounds.

    Parameters:
    boxes (np.array): Tensor of bounding boxes with shape (N, 4)

    Returns:
    np.array: Adjusted bounding boxes
    """

    adjusted_boxes = boxes.copy()
    n_boxes = boxes.shape[0]

    for i in range(n_boxes):
        for j in range(i+1, n_boxes):
            # compute overlap area
            area_overlap = calculate_bbox_overlap_area(adjusted_boxes[i], adjusted_boxes[j])

            # compute proportion of smaller box covered by overlap
            box1_area = (adjusted_boxes[i][2] - adjusted_boxes[i][0]) * (adjusted_boxes[i][3] - adjusted_boxes[i][1])
            box2_area = (adjusted_boxes[j][2] - adjusted_boxes[j][0]) * (adjusted_boxes[j][3] - adjusted_boxes[j][1])
            smaller_box_area = min(box1_area, box2_area)
            overlap_ratio = area_overlap / smaller_box_area

            # if overlap is significant wrt smaller box, adjust boxes
            if area_overlap > 0 and overlap_ratio > overlap_ratio_threshold:
                if box1_area > box2_area:
                    larger_box = adjusted_boxes[i]
                    smaller_box = adjusted_boxes[j]
                else:
                    larger_box = adjusted_boxes[j]
                    smaller_box = adjusted_boxes[i]

                xmin_l, ymin_l, xmax_l, ymax_l = larger_box
                xmin_s, ymin_s, xmax_s, ymax_s = smaller_box

                change_ymax = False
                change_ymin = False
                if ymin_l < ymin_s < ymax_l:
                    candidate_ymax_l = ymin_s
                    change_ymax = True
                if ymin_l < ymax_s < ymax_l:
                    candidate_ymin_l = ymax_s
                    change_ymin = True

                if change_ymax and change_ymin:
                    if abs(candidate_ymax_l - ymax_l) < abs(candidate_ymin_l - ymin_l):
                        ymax_l = candidate_ymax_l
                    else:
                        ymin_l = candidate_ymin_l
                elif change_ymax:
                    ymax_l = candidate_ymax_l
                elif change_ymin:
                    ymin_l = candidate_ymin_l
                else:
                    raise ValueError("This should not happen")

                larger_box[:] = [xmin_l, ymin_l, xmax_l, ymax_l]
    return adjusted_boxes

def calculate_bbox_overlap_area(
    bbox1: Union[np.ndarray, List[float]], 
    bbox2: Union[np.ndarray, List[float]]
) -> float:
    """
    Calculate the overlapping area between two bounding boxes.
    
    Parameters:
    box1, box2 (list or np.array): Bounding boxes in the format [xmin, ymin, xmax, ymax]
    
    Returns:
    float: Overlapping area between the two bounding boxes
    """
    xmin_1, ymin_1, xmax_1, ymax_1 = bbox1
    xmin_2, ymin_2, xmax_2, ymax_2 = bbox2
    
    # calculate intersection coordinates
    xmin = max(xmin_1, xmin_2)
    ymin = max(ymin_1, ymin_2)
    xmax = min(xmax_1, xmax_2)
    ymax = min(ymax_1, ymax_2)
    
    # calculate intersection width and height
    width = max(0, xmax - xmin)
    height = max(0, ymax - ymin)
    
    # calculate intersection area
    overlap_area = width * height
    
    return overlap_area

def inflate_bboxes(
    bboxes: np.ndarray, 
    n_pix_vertical: Optional[int] = None, 
    n_pix_horizontal: Optional[int] = None
) -> np.ndarray:
    """
    Inflate bounding boxes by a certain number of pixels on each side.

    Args:
    bboxes: np.ndarray of shape (N, 4) where each row is (xmin, ymin, xmax, ymax).
    n_pix_vertical: Number of pixels to inflate the bounding boxes vertically on either side.
    n_pix_horizontal: Number of pixels to inflate the bounding boxes horizontally on either side.

    Returns:
    np.ndarray: Inflated bounding boxes.
    """

    bboxes = bboxes.copy()

    if n_pix_vertical is not None:
        bboxes[:, 1] -= n_pix_vertical
        bboxes[:, 3] += n_pix_vertical

    if n_pix_horizontal is not None:
        bboxes[:, 0] -= n_pix_horizontal
        bboxes[:, 2] += n_pix_horizontal

    return bboxes

def get_image_foreground(
    image: Union[np.ndarray, Image.Image], 
    depth: np.ndarray, 
    distance_threshold: float = 3, 
    bg_color: int = 255
) -> np.ndarray:
    """
    Simple function to get the foreground of an image using the corresponding depth mask.
    All points in the depth mask that are further than the distance threshold are considered background.

    Args:
    image: np.ndarray of shape (H, W, 3) representing an RGB image.
    depth: np.ndarray of shape (H, W) representing the depth mask.
    bg_color: int representing the color to use for the background for every channel.

    Returns:
    np.ndarray: Image with the background removed.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    masked_image = image.copy()
    masked_image[depth > distance_threshold, :] = bg_color
    masked_image[np.isnan(depth), :] = bg_color

    return masked_image

def show_detections(
    image: Union[Image.Image, np.ndarray], 
    detections: List[str], 
    bboxes: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Display an image with bounding boxes and detection labels.

    Args:
        image (Union[Image.Image, np.ndarray]): The input image, either as a PIL Image or a NumPy array.
        detections (List[str]): A list of N detection labels corresponding to the bounding boxes.
        bboxes (np.ndarray): An N x 4 array of bounding boxes, where each bounding box is represented as [x_min, y_min, x_max, y_max].
        save_path (Optional[str]): Path to save the image to.

    Returns:
        None
    """
    fig, ax = plt.subplots(ncols=1, squeeze=False, figsize=(12.8, 9.6))

    if len(detections) > 0:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(image, Image.Image):
            tensor_image = F.pil_to_tensor(image)
        image_with_bboxes = draw_bounding_boxes(tensor_image, torch.tensor(bboxes), labels=detections, width=5)
        image_with_bboxes = image_with_bboxes.detach()
        image_with_bboxes = F.to_pil_image(image_with_bboxes)    
        ax[0, 0].imshow(np.asarray(image_with_bboxes))
        ax[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    else:
        ax[0, 0].imshow(np.asarray(image))
        ax[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.clf()
        plt.close(fig)
    else:
        plt.show()