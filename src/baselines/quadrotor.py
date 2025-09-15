import copy
import random

from pathlib import Path

import base64
import cv2
import io
import matplotlib.pyplot as plt
import numpy as np

from openai import OpenAI
from PIL import Image

from api_key import OPENAI_API_KEY
from src.baselines.prompts import (
    parse_approach_path_response, parse_landing_position_response, 
    APPROACH_PATH_QUERY_TEMPLATE, SYSTEM_MESSAGE, LANDING_POSITION_QUERY_TEMPLATE
)
from src.opspace.trajectories.bezier import BezierCurve
from src.splatting.datasets import IsaacDataset
from src.utils.io import create_directory, load_yaml, save_dict_to_pkl, save_dict_to_json
from src.utils.rgbd import depth_to_pointcloud, get_normals
from src.utils.llm import query_llm
from src.vision_models.detection_models import OWLv2
from src.vision_models.utils.detection import get_image_foreground, inflate_bboxes, show_detections

def draw_numbered_circle(
    image,
    center,
    radius,
    circle_color=(200, 200, 200),
    text_color=(0, 0, 0),
    number=1,
    alpha=0.65,
    font_scale=0.5,
    thickness=1
):
    """
    Draw a partially transparent circle with a centered number on an image.

    Args:
        image (np.ndarray): The image to draw on (H x W x C).
        center (tuple): (x, y) center of the circle.
        radius (int): Circle radius.
        circle_color (tuple): RGB color of the circle.
        text_color (tuple): RGB color of the text.
        number (int): The digit to draw in the circle.
        alpha (float): Transparency factor for the circle.
        font_scale (float): Font scale for the number.
        thickness (int): Thickness for the number text.

    Returns:
        np.ndarray: Image with the circle and number drawn.
    """
    # Convert colors to BGR for OpenCV
    circle_color_bgr = tuple(reversed(circle_color))
    text_color_bgr = tuple(reversed(text_color))

    overlay = image.copy()
    cv2.circle(overlay, center, radius, circle_color_bgr, thickness=-1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    text = str(number)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color_bgr, thickness, lineType=cv2.LINE_AA)
    return image

def draw_colored_curve_with_number(
    image,
    points,
    number,
    circle_radius=12,
    line_color=(200, 200, 200),
    text_color=(0, 0, 0),
    alpha=0.65,
    font_scale=0.5,
    font_thickness=1,
    line_thickness=2,
    annotate_fraction=0.9
):
    """
    Draw a polyline (curve) and a numbered circle at a specified fraction along the curve.

    Args:
        image (np.ndarray): The image to draw on (H x W x C).
        points (np.ndarray): Array of shape (N, 2) with pixel coordinates.
        number (int): The digit to draw in the circle.
        circle_radius (int): Circle radius.
        line_color (tuple): RGB color of the line and circle.
        text_color (tuple): RGB color of the text.
        alpha (float): Transparency for the circle.
        font_scale (float): Font scale for the number.
        thickness (int): Line and text thickness.
        annotate_fraction (float): Where to place the circle (0=start, 1=end).

    Returns:
        np.ndarray: Image with the curve and number drawn.
    """
    # Draw the curve
    color_bgr = tuple(reversed(line_color))
    pts = points.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=False, color=color_bgr, thickness=line_thickness)

    # Annotate with a numbered circle at a specified fraction along the curve
    if len(points) >= 2:
        idx_annotate = int(annotate_fraction * (len(points) - 1))
        center = tuple(points[idx_annotate])
        image = draw_numbered_circle(
            image,
            center=center,
            radius=circle_radius,
            circle_color=line_color,
            text_color=text_color,
            number=number,
            alpha=alpha,
            font_scale=font_scale,
            thickness=font_thickness
        )
    return image



class VisualPromptLander:
    def __init__(self, config):
        self.config = config

        # load dataset
        self.dataset = None
        self._load_dataset()

        # retrieve the landing target
        self.landing_target = None
        self._set_landing_target()

        # load detection model
        self.detection_model = None
        self._load_detection_model()

        # set up OpenAI client
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        # set up output directory
        run_name = self.config["dataset_config"]["dataset_path"].split("/")[-1] 
        self.output_directory = Path(self.config["save_directory"]) / run_name / "visual_prompt_baseline"
        create_directory(self.output_directory, overwrite=True)

    def _load_detection_model(self):
        detection_model_config = copy.deepcopy(self.config["preprocessing"]["detection_model"])
        detection_model_config["texts"] = [self.landing_target]
        self.detection_model = OWLv2(detection_model_config)

    def _load_dataset(self):
        # load dataset for baseline
        dataset_path = self.config["dataset_config"]["dataset_path"]
        dataset_path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
        config = copy.deepcopy(self.config)
        config["dataset_config"]["dataset_path"] = dataset_path / "baseline"
        self.dataset = IsaacDataset(config["dataset_config"])

    def _set_landing_target(self):
        dataset_path = self.config["dataset_config"]["dataset_path"]
        dataset_path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
        scenario_config_path = dataset_path / "scenario_config.yaml"
        scenario_config = load_yaml(scenario_config_path)
        self.landing_target = scenario_config["task"]["landing_target"]

    def land(self, case_idx):
        pass

    def annotate_landing_candidates(self,
        image, bbox=None, mask=None, num_samples=15, radius=8,
        circle_color=(200, 200, 200),  # RGB
        text_color=(0, 0, 0),   # RGB
        seed=None, oversample_factor=5, max_attempts=100
    ):
        """
        Annotate non-overlapping sampled pixels with numbered circles.
        You can specify either a bbox or a mask. If both are given, mask is used.

        Args:
            image (np.ndarray): Input image (H x W x C).
            bbox (tuple): (x_min, y_min, x_max, y_max).
            mask (np.ndarray): Boolean mask (H x W) for valid sampling locations.
            num_samples (int): Number of points to sample.
            radius (int): Circle radius.
            ...
        Returns:
            np.ndarray: Annotated image.
            dict: Mapping from digit to (x, y) pixel coordinates.
        """
        circle_color_bgr = tuple(reversed(circle_color))
        text_color_bgr = tuple(reversed(text_color))

        if seed is not None:
            np.random.seed(seed)

        h, w = image.shape[:2]
        annotated = image.copy()
        selected_points = []

        # get candidate pool
        if mask is not None:
            ys, xs = np.where(mask)
            valid_coords = np.stack([xs, ys], axis=1)
            # remove border points that can't fit a circle
            valid = (
                (xs >= radius) & (xs < w - radius) &
                (ys >= radius) & (ys < h - radius)
            )
            valid_coords = valid_coords[valid]
        elif bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            x_min = max(0, min(x_min, w - 1)) + radius
            x_max = max(0, min(x_max, w)) - radius
            y_min = max(0, min(y_min, h - 1)) + radius
            y_max = max(0, min(y_max, h)) - radius
            xs, ys = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
            valid_coords = np.stack([xs.ravel(), ys.ravel()], axis=1)
        else:
            raise ValueError("Either bbox or mask must be provided.")

        if len(valid_coords) == 0:
            raise ValueError("No valid sampling locations found.")

        # sample points
        for _ in range(max_attempts):
            if len(valid_coords) < num_samples:
                print("Warning: Not enough valid locations for requested samples.")
                break
            idxs = np.random.choice(len(valid_coords), size=min(num_samples * oversample_factor, len(valid_coords)), replace=False)
            candidates = valid_coords[idxs]

            for p in candidates:
                if len(selected_points) == 0:
                    selected_points.append(p)
                else:
                    dists = np.linalg.norm(np.array(selected_points) - p, axis=1)
                    if np.all(dists >= 2 * radius):
                        selected_points.append(p)
                if len(selected_points) == num_samples:
                    break
            if len(selected_points) == num_samples:
                break

        if len(selected_points) < num_samples:
            print(f"Warning: Only placed {len(selected_points)} of {num_samples} circles.")

        # draw on image
        digit_to_pixel = {}
        alpha = 0.65
        for i, (x, y) in enumerate(selected_points, 1):
            center = (int(x), int(y))
            annotated = draw_numbered_circle(
                annotated,
                center=center,
                radius=radius,
                circle_color=circle_color,
                text_color=text_color,
                number=i,
                alpha=0.65,
                font_scale=0.5,
                thickness=1
            )
            # overlay = annotated.copy()
            # cv2.circle(overlay, center, radius, circle_color_bgr, thickness=-1)
            # cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
            # text = str(i)
            # font_scale = 0.5
            # thickness = 1
            # text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # text_x = center[0] - text_size[0] // 2
            # text_y = center[1] + text_size[1] // 2
            # cv2.putText(annotated, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            #             font_scale, text_color_bgr, thickness, lineType=cv2.LINE_AA)
            digit_to_pixel[i] = np.array([int(x), int(y)])

        return annotated, digit_to_pixel

    def determine_landing_position(self, case_idx):
        # get color image, depth, camera pose
        color_image = self.dataset.get_color_image(case_idx)
        depth_image = self.dataset.get_depth_image(case_idx)
        T_c2w = self.dataset.get_camera_pose(case_idx)
        camera_position = T_c2w[:3, 3]

        # search for the landing target with detection model
        # detection_query = get_image_foreground(color_image, depth_image, distance_threshold=3, bg_color=255)
        # detections, bboxes, image = self.detection_model.detect(detection_query)
        # # draw the bounding boxes on the image
        # bboxes = inflate_bboxes(bboxes, n_pix_vertical=10, n_pix_horizontal=10)
        # if len(detections) != 1:
        #     raise
        # # convert color image to PIL Image
        # # from PIL import Image
        # # color_image_pil = Image.fromarray(color_image)
        # # color_image = color_image.astype(np.uint8)
        # show_detections(color_image, detections, bboxes)

        # determine valid sampling locations
        normals = get_normals(depth_image, self.dataset.K, T_c2w)
        point_cloud = depth_to_pointcloud(depth_image, self.dataset.K, T_c2w).reshape(*normals.shape)

        # mask pixels with normals pointing close to up (z-axis) and above z threshold
        dot_product_threshold = 0.9
        height_threshold = 0.4
        camera_proximity_threshold = 5.0 # meters

        vertical_dot_products = normals @ np.array([0, 0, 1])
        distances_to_camera = np.linalg.norm(point_cloud - camera_position, axis=2)

        landing_site_mask = (vertical_dot_products > dot_product_threshold) & \
            (point_cloud[:, :, 2] > height_threshold) & \
            (distances_to_camera < camera_proximity_threshold)


        # show color image, but only pixels under mask
        color_image_masked = color_image.copy().astype(np.uint8)
        color_image_masked[~landing_site_mask] = 0

        annotated_image, digit_to_pixel = self.annotate_landing_candidates(
            image=color_image,
            mask=landing_site_mask,
        )

        # query the LLM
        system_message = SYSTEM_MESSAGE
        query_message = LANDING_POSITION_QUERY_TEMPLATE.format(landing_target=self.landing_target)

        img_byte_arr = io.BytesIO()
        pil_img = Image.fromarray(annotated_image)
        pil_img.save(img_byte_arr, format="jpeg")
        img_byte_arr.seek(0)
        img_bytes = img_byte_arr.getvalue()
        base64_img = base64.b64encode(img_bytes).decode("utf-8")

        response = query_llm(
            self.client,
            system_message=system_message,
            user_message=query_message,
            user_images=[base64_img],
        )

        # parse the response and get the landing position
        landing_id = parse_landing_position_response(response.output_text)
        landing_pixel_coords = digit_to_pixel[landing_id]
        landing_position = point_cloud[landing_pixel_coords[1], landing_pixel_coords[0]]

        # save data
        # save annotated image
        annotated_image_path = self.output_directory / f"landing_position_candidates_{case_idx}.png"
        cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR, dst=annotated_image)
        cv2.imwrite(str(annotated_image_path), annotated_image)

        # save image with only the landing position annotated
        landing_position_image_path = self.output_directory / f"selected_landing_position_{case_idx}.png"
        landing_position_image = color_image.copy()
        landing_position_image = draw_numbered_circle(
            landing_position_image,
            center=landing_pixel_coords,
            radius=8,
            circle_color=(0, 255, 0),  # RGB
            text_color=(0, 0, 0),   # RGB
            number=landing_id,
            alpha=0.65,
            font_scale=0.5,
            thickness=1
        )
        cv2.cvtColor(landing_position_image, cv2.COLOR_RGB2BGR, dst=landing_position_image)
        cv2.imwrite(str(landing_position_image_path), landing_position_image)

        # save llm response
        response_path = self.output_directory / f"landing_position_response_{case_idx}.pkl"
        save_dict_to_pkl(
            response,
            response_path,
            directory=self.output_directory,
        )

        return landing_position
    
    def annotate_approach_path_candidates(
        self,
        image,
        curves_3d,           # List of (N_i, 3) arrays, each a curve in world coordinates
        T_c2w,               # 4x4 camera-to-world pose
        K,                   # 3x3 camera intrinsics
        radius=12,
        line_color=None,
        circle_color=(200, 200, 200),
        text_color=(0, 0, 0),
        alpha=0.65,
        font_scale=0.5,
        thickness=2
    ):
        """
        Annotate projected 3D curves on the image, with a numbered circle for each curve.

        Args:
            image (np.ndarray): Input image (H x W x C).
            curves_3d (list of np.ndarray): Each (N_i, 3) array is a curve in world coordinates.
            T_c2w (np.ndarray): 4x4 camera-to-world pose.
            K (np.ndarray): 3x3 camera intrinsics.
            radius (int): Circle radius.
            circle_color (tuple): RGB color of the circle.
            text_color (tuple): RGB color of the text.
            alpha (float): Transparency for the circle.
            font_scale (float): Font scale for the number.
            thickness (int): Thickness for the number text.

        Returns:
            np.ndarray: Annotated image.
            dict: Mapping from digit to curve index.
        """
        annotated = image.copy()
        T_w2c = np.linalg.inv(T_c2w)
        digit_to_curve = {}

        # Use matplotlib colormap for distinct colors
        if line_color is None:
            n_curves = len(curves_3d)
            cmap = plt.get_cmap('tab10') if n_curves <= 10 else plt.get_cmap('hsv')
            colors = [tuple(int(255 * c) for c in cmap(i % cmap.N)[:3]) for i in range(n_curves)]

        for idx, curve in enumerate(curves_3d, 1):
            # Project curve to camera coordinates
            curve_h = np.concatenate([curve, np.ones((curve.shape[0], 1))], axis=1)  # (N, 4)
            # curve_cam = (T_w2c @ curve_h.T).T[:, :3]  # (N, 3)
            # valid = curve_cam[:, 2] > 0
            # if not np.any(valid):
            #     continue
            # curve_cam = curve_cam[valid]
            # # Project to image
            # curve_img = (K @ curve_cam.T).T
            # curve_img = curve_img[:, :2] / curve_img[:, 2:3]
            # curve_img = np.round(curve_img).astype(int)

            curve_cam = (T_w2c @ curve_h.T).T[:, :3]  # (N, 3)
            valid_z = curve_cam[:, 2] > 0
            curve_cam = curve_cam[valid_z]
            if len(curve_cam) == 0:
                continue
            curve_img = (K @ curve_cam.T).T
            curve_img = curve_img[:, :2] / curve_img[:, 2:3]
            curve_img = np.round(curve_img).astype(int)

            # Now filter for points inside the image
            h, w = image.shape[:2]
            inside = (
                (curve_img[:, 0] >= 0) & (curve_img[:, 0] < w) &
                (curve_img[:, 1] >= 0) & (curve_img[:, 1] < h)
            )
            curve_img_visible = curve_img[inside]
            if len(curve_img_visible) < 2:
                continue

            # Draw the curve
            color = colors[idx - 1] if line_color is None else line_color
            annotated = draw_colored_curve_with_number(
                annotated,
                points=curve_img_visible,
                number=idx,
                circle_radius=radius,
                line_color=color,
                text_color=text_color,
                alpha=alpha,
                font_scale=font_scale,
                line_thickness=thickness,
                annotate_fraction=0.5
            )

            digit_to_curve[idx] = idx - 1  # digit: curve index

        return annotated, digit_to_curve
    
    def determine_approach_trajectory(self, case_idx, landing_position):
        # get color image, depth, camera pose
        color_image = self.dataset.get_color_image(case_idx)
        depth_image = self.dataset.get_depth_image(case_idx)
        T_c2w = self.dataset.get_camera_pose(case_idx)
        camera_position = T_c2w[:3, 3]
        print(f"Camera position: {camera_position}")
        
        # sample bezier curves from initial position to landing position
        approach_beziers = []
        approach_paths = []
        approach_waypoints = []
        
        # set initial point to 0.15 meters in front of the camera
        camera_offset = np.array([0.0, 0.0, 0.01, 1.0]) # np.array([0.0, 0.0, 0.15, 1.0])
        initial_position = (T_c2w @ camera_offset)[:3] # camera_position + 
        initial_position[2] = landing_position[2] # set height to that of the landing position

        # generate curve for each approach angle
        approach_vector = -(landing_position - initial_position)
        approach_vector /= np.linalg.norm(approach_vector)
        approach_radius = 2.25
        approach_angle_limit = np.deg2rad(135)
        approach_angles = np.linspace(approach_angle_limit, -approach_angle_limit, num=9)
        for angle in approach_angles:
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ]).T
            approach_waypoint = landing_position + rotation_matrix @ (approach_radius * approach_vector)
        
            # create Bezier curve
            approach_bezier = BezierCurve(
                np.array([initial_position, approach_waypoint, landing_position]),
                a=0.0,
                b=1.0,
            )
            approach_path = approach_bezier(np.linspace(0, 1, num=100))
            # approach_path[:, 2] += 0.25
            approach_beziers.append(approach_bezier)
            approach_paths.append(approach_path)
            approach_waypoints.append(approach_waypoint)

        # create image with annotated approach paths
        annotated_image, digit_to_curve = self.annotate_approach_path_candidates(
            image=color_image,
            curves_3d=approach_paths,  # list of np.ndarray, each (N, 3)
            T_c2w=T_c2w,
            K=self.dataset.K,
        )

        # query the LLM
        system_message = SYSTEM_MESSAGE
        query_message = APPROACH_PATH_QUERY_TEMPLATE.format(landing_target=self.landing_target)

        img_byte_arr = io.BytesIO()
        pil_img = Image.fromarray(annotated_image)
        pil_img.save(img_byte_arr, format="jpeg")
        img_byte_arr.seek(0)
        img_bytes = img_byte_arr.getvalue()
        base64_img = base64.b64encode(img_bytes).decode("utf-8")

        response = query_llm(
            self.client,
            system_message=system_message,
            user_message=query_message,
            user_images=[base64_img],
        )

        # parse the response and get the approach path
        approach_id = parse_approach_path_response(response.output_text)
        approach_path = approach_paths[digit_to_curve[approach_id]]

        # save data
        # save annotated image
        annotated_image_path = self.output_directory / f"approach_path_candidates_{case_idx}.png"
        cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR, dst=annotated_image)
        cv2.imwrite(str(annotated_image_path), annotated_image)

        # save image with only the landing position annotated
        approach_path_image_path = self.output_directory / f"selected_approach_path_{case_idx}.png"
        approach_path_image = color_image.copy()

        # project curve to camera coordinates
        curve_h = np.concatenate([approach_path, np.ones((approach_path.shape[0], 1))], axis=1)  # (N, 4)
        curve_cam = (np.linalg.inv(T_c2w) @ curve_h.T).T[:, :3]  # (N, 3)
        valid = curve_cam[:, 2] > 0
        curve_cam = curve_cam[valid]
        # project to image
        curve_img = (self.dataset.K @ curve_cam.T).T
        curve_img = curve_img[:, :2] / curve_img[:, 2:3]
        curve_img = np.round(curve_img).astype(int)


        approach_path_image = draw_colored_curve_with_number(
            approach_path_image,
            points=curve_img,
            number=approach_id,
            line_color=(0, 255, 0)
        )
        cv2.cvtColor(approach_path_image, cv2.COLOR_RGB2BGR, dst=approach_path_image)
        cv2.imwrite(str(approach_path_image_path), approach_path_image)

        # save llm response
        response_path = self.output_directory / f"approach_path_response_{case_idx}.pkl"
        save_dict_to_pkl(
            response,
            response_path,
            directory=self.output_directory,
        )

        # save overall action specification
        action_dict = {
            "initial_position": initial_position[:2].tolist(),
            "approach_waypoint": approach_waypoints[digit_to_curve[approach_id]][:2].tolist(),
            "landing_position": landing_position[:2].tolist(),
            "landing_target_height": landing_position[2],
        }
        save_dict_to_json(action_dict, f"action_specification_{case_idx}.json", directory=self.output_directory)

        return approach_path

    def run(self, case_idx):
        landing_position = self.determine_landing_position(case_idx)
        approach_trajectory = self.determine_approach_trajectory(case_idx, landing_position)

        return landing_position, approach_trajectory


if __name__ == "__main__":
    import yaml

    config_path = "/home/anonymous/Documents/Research/gaussian-splatting-playground/config/quadrotor_scene_1.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    lander = VisualPromptLander(config)
    lander.run(0)
