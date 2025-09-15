import copy

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from src.splatting.datasets import get_dataset
from src.utils.io import create_directory
from src.utils.segmentation_2d import refine_segmentation
from src.vision_models.detection_models import OWLv2
from src.vision_models.segmentation_models import SAM2
from src.vision_models.utils.detection import adjust_bboxes, eliminate_bbox_overlap, inflate_bboxes, get_image_foreground, show_detections
from src.vision_models.utils.segmentation import get_background_mask, get_ground_mask, fill_holes, show_mask

class DataPreprocessor:
    def __init__(self, preprocessing_config, dataset_config):
        self.preprocessing_config = preprocessing_config

        # load dataset
        self.dataset_config = copy.deepcopy(dataset_config)
        self.dataset_config["downsample"] = False
        self.dataset = get_dataset(self.dataset_config["dataset"])(self.dataset_config)

        # load vision models
        self.detection_model = OWLv2(self.preprocessing_config["detection_model"])
        self.segmentation_model = SAM2(self.preprocessing_config["segmentation_model"])

        # set data directory
        data_directory = self.dataset_config["dataset_path"]
        self.data_directory = Path(data_directory) if isinstance(data_directory, str) else data_directory

    def run(self):
        if self.preprocessing_config["segmentation_model"]["video_mode"]:
            self._run_video_segmentation()
        else:
            self._run_image_segmentation()

    def _run_video_segmentation(self):
        # create directories for outputs
        create_directory(self.data_directory / "instance_segmentation", overwrite=True)
        create_directory(self.data_directory / "bboxes", overwrite=True)

        if self.preprocessing_config["save_debug_images"]:
            create_directory(self.data_directory / "debugging" / "bboxes", overwrite=True)
            create_directory(self.data_directory / "debugging" / "adjusted_bboxes", overwrite=True)
            create_directory(self.data_directory / "debugging" / "inflated_bboxes", overwrite=True)
            create_directory(self.data_directory / "debugging" / "segmentation", overwrite=True)
            create_directory(self.data_directory / "debugging" / "segmentation_overlay", overwrite=True)

        # detections from first frame
        K = self.dataset.K
        T_c2w = self.dataset.get_camera_pose(0)
        color = self.dataset.get_color_image(0)
        depth = self.dataset.get_depth_image(0)
        detections, boxes = self._get_detections(0, color, depth, T_c2w)

        # prompt video segmentation with these detections
        video_segmentation = self.segmentation_model.segment_video(str(self.data_directory / "rgb_jpg"), box=boxes)

        for idx in range(len(self.dataset)):
            # get frame from dataset
            T_c2w = self.dataset.get_camera_pose(idx)
            color = self.dataset.get_color_image(idx)
            depth = self.dataset.get_depth_image(idx)
            frame_segmentation = video_segmentation[idx]

            # 2) segmentation step: generates instance segmentation masks for the detected objects
            segmentation = self._complete_video_frame_segmentation(idx, color, depth, T_c2w, K, frame_segmentation)

            # 3) save results
            if idx == 0:
                self._save_results(idx, detections, segmentation, bboxes=boxes)
            else:
                self._save_results(idx, detections, segmentation)

    def _complete_video_frame_segmentation(self, idx, color, depth, T_c2w, K, frame_segmentation):
        # segment ground
        ground_mask = get_ground_mask(depth, T_c2w, K)

        # segment objects previously detected
        foreground_masks = np.concatenate(
            [frame_segmentation[i] for i in sorted(frame_segmentation.keys())],
            axis=0
        )

        # segment background
        if len(foreground_masks.shape) > 3:
            foreground_masks = foreground_masks.squeeze()
        mask_array = np.concatenate((ground_mask[None], foreground_masks), axis=0)
        background_mask = get_background_mask(mask_array, depth, T_c2w, K)

        # assemble segmentation masks
        segmentation = -1 * np.ones_like(depth)
        masks = [mask.squeeze().astype(bool) for mask in foreground_masks] + [ground_mask, background_mask]
        for i, mask in enumerate(masks):
            segmentation[mask] = int(i)

        # fill holes
        fill_mask = (segmentation == -1)
        segmentation = fill_holes(segmentation, fill_mask)
        segmentation[depth == np.nan] = len(masks) # unknown class

        segmentation[segmentation > len(masks)] = len(masks) # unknown class
        segmentation = segmentation.astype(np.uint8)

        # refine
        refined_segmentation = refine_segmentation(segmentation, depth, K, T_c2w, edge_dilation=7, k_knn=5)

        ground_label = np.unique(refined_segmentation)[-2]
        changed_to_ground = (refined_segmentation == ground_label) & (segmentation != ground_label)
        refined_segmentation[changed_to_ground] = segmentation[changed_to_ground]

        # save debug images
        if self.preprocessing_config["save_debug_images"]:
            plt.figure(figsize=(10, 10), dpi=200)
            plt.imshow(refined_segmentation, interpolation="none")
            plt.axis('off')
            plt.savefig(fname=str(self.data_directory / "debugging" / "segmentation" / f"{idx}.png"))
            plt.clf()
            plt.close()

            plt.figure(figsize=(10, 10))
            plt.imshow(color)
            for val in np.unique(refined_segmentation):
                mask = (refined_segmentation == val)
                show_mask(mask, plt.gca(), random_color=True)
            plt.axis('off')
            plt.savefig(fname=str(self.data_directory / "debugging" / "segmentation_overlay" / f"{idx}.png"))
            plt.clf()
            plt.close()

        return refined_segmentation

    def _run_image_segmentation(self):
        # create directories for outputs
        create_directory(self.data_directory / "instance_segmentation", overwrite=True)

        if self.preprocessing_config["save_debug_images"]:
            create_directory(self.data_directory / "debugging" / "bboxes", overwrite=True)
            create_directory(self.data_directory / "debugging" / "adjusted_bboxes", overwrite=True)
            create_directory(self.data_directory / "debugging" / "inflated_bboxes", overwrite=True)
            create_directory(self.data_directory / "debugging" / "segmentation", overwrite=True)
            create_directory(self.data_directory / "debugging" / "segmentation_overlay", overwrite=True)


        K = self.dataset.K
        for idx in range(len(self.dataset)):
            # get frame from dataset
            T_c2w = self.dataset.get_camera_pose(idx)
            color = self.dataset.get_color_image(idx)
            depth = self.dataset.get_depth_image(idx)


            # 1) detection step: generates bounding box prompts for the segmentation model
            detections, boxes = self._get_detections(idx, color, depth, T_c2w)

            # 2) segmentation step: generates instance segmentation masks for the detected objects
            segmentation = self._get_image_segmentation(idx, color, depth, T_c2w, K, boxes)

            # 3) save segmentation results
            self._save_results(idx, detections, segmentation, bboxes=boxes)

    def _get_detections(self, idx, color, depth, T_c2w):
        # removing background can help reduce noise, identify foreground objects better
        if self.preprocessing_config["detection"]["mask_image_background"]:
            detection_query = get_image_foreground(color, depth, distance_threshold=5, bg_color=255)
        else:
            detection_query = color

        # query detection model
        detections, boxes, detection_image = self.detection_model.detect(detection_query, verbose=False)

        # remove bounding boxes' overlap, useful to prevent small objects being contained in larger objects' bounding boxes
        if self.preprocessing_config["detection"]["eliminate_overlap"]:
            boxes, eliminated_indices = eliminate_bbox_overlap(boxes)
            detections = [detections[i] for i in range(len(detections)) if i not in eliminated_indices]
            nonoverlap_boxes = boxes

        # remove bounding boxes' overlap, useful to prevent small objects being contained in larger objects' bounding boxes
        if self.preprocessing_config["detection"]["adjust_bbox_overlap"]:
            boxes = adjust_bboxes(boxes)
            adjusted_boxes = boxes

        # inflate bounding boxes, useful when boxes don't completely enclose entities or segmentation model undersegments objects
        if self.preprocessing_config["detection"]["inflate_bboxes"]:
            boxes = inflate_bboxes(boxes, n_pix_vertical=5, n_pix_horizontal=10)
            inflated_boxes = boxes

        # save debug images
        if self.preprocessing_config["save_debug_images"]:    
            show_detections(detection_image, detections, boxes, save_path=self.data_directory / "debugging" / "bboxes" / f"{idx}.png")
            if self.preprocessing_config["detection"]["adjust_bbox_overlap"]:
                show_detections(detection_image, detections, adjusted_boxes, save_path=self.data_directory / "debugging" / "adjusted_bboxes" / f"{idx}.png")
            if self.preprocessing_config["detection"]["inflate_bboxes"]:
                show_detections(detection_image, detections, inflated_boxes, save_path=self.data_directory / "debugging" / "inflated_bboxes" / f"{idx}.png")

        return detections, boxes

    def _get_image_segmentation(self, idx, color, depth, T_c2w, K, boxes):
        # segment ground
        ground_mask = get_ground_mask(depth, T_c2w, K, ground_height=0.02)

        # segment objects previously detected
        foreground_masks, scores = self.segmentation_model.segment_image(color, box=boxes)

        # segment background
        if len(foreground_masks.shape) > 3:
            foreground_masks = foreground_masks.squeeze()
        mask_array = np.concatenate((ground_mask[None], foreground_masks), axis=0)
        background_mask = get_background_mask(mask_array, depth, T_c2w, K)

        # assemble segmentation masks
        segmentation = -1 * np.ones_like(depth)
        masks = [mask.squeeze().astype(bool) for mask in foreground_masks] + [ground_mask, background_mask]
        for i, mask in enumerate(masks):
            segmentation[mask] = int(i)

        # fill holes
        fill_mask = (segmentation == -1)
        segmentation = fill_holes(segmentation, fill_mask)
        segmentation[depth == np.nan] = len(masks) # unknown class

        segmentation[segmentation > len(masks)] = len(masks) # unknown class
        segmentation = segmentation.astype(np.uint8)

        # save debug images
        if self.preprocessing_config["save_debug_images"]:
            plt.figure(figsize=(10, 10))
            plt.imshow(segmentation)
            plt.axis('off')
            plt.savefig(fname=str(self.data_directory / "debugging" / "segmentation" / f"{idx}.png"))
            plt.clf()
            plt.close()

            plt.figure(figsize=(10, 10))
            plt.imshow(color)
            for val in np.unique(segmentation):
                mask = (segmentation == val)
                show_mask(mask, plt.gca(), random_color=True)
            plt.axis('off')
            plt.savefig(fname=str(self.data_directory / "debugging" / "segmentation_overlay" / f"{idx}.png"))
            plt.clf()
            plt.close()

        return segmentation
    
    def _save_results(self, idx, detections, segmentation, bboxes=None):
        # prepare data output
        segmented_frame = {
            "segmented_image": None,
            "annotations": None,
            "ids": None
        }

        # record annotations        
        annotations = []
        ids = []
        for i in range(len(detections)):
            annotations.append(detections[i])
            ids.append(i)
        annotations.append("the ground")
        annotations.append("the background")
        ids.append(len(detections))
        ids.append(len(detections) + 1)
        if len(np.unique(segmentation)) > ids[-1] + 1:
            annotations.append("unknown")
            ids.append(len(detections) + 2)

        segmented_frame["segmented_image"] = segmentation
        segmented_frame["annotations"] = annotations
        segmented_frame["ids"] = ids

        # save results
        with open(self.data_directory / "instance_segmentation" / f"{idx:04d}.pkl", 'wb') as f:
            pickle.dump(segmented_frame, f)

        if bboxes is not None:
            np.save(self.data_directory / "bboxes" / f"{idx:04d}.npy", bboxes)