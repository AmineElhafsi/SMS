import math
import os

from pathlib import Path

import cv2
import numpy as np
import torch

from scipy.spatial.transform import Rotation as R

from src.utils.io import load_pkl

class SegmentedFrame:
    def __init__(self, segmented_image, annotations, ids):
        self.segmented_image = segmented_image
        self.annotations = annotations
        self.ids = ids

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config: dict) -> None:
        self.config = config

        self.dataset_path = Path(config["dataset_path"])
        self.frame_limit = config.get("frame_limit", -1)

        self.image_height = config["H"]
        self.image_width = config["W"]
        self.fx = config["fx"]
        self.fy = config["fy"]
        self.cx = config["cx"]
        self.cy = config["cy"]
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]).astype(np.float32) # camera intrinsics matrix

        self.downsample = config["downsample"]
        if self.downsample:
            self.image_height //= 2
            self.image_width //= 2
            self.fx /= 2
            self.fy /= 2
            self.cx /= 2
            self.cy /= 2
            self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]).astype(np.float32) # camera intrinsics matrix
        
        self.depth_scale = config["depth_scale"]
        self.distortion = np.array(config['distortion']) if 'distortion' in config else None

        self.crop_edge = config['crop_edge'] if 'crop_edge' in config else 0
        if self.crop_edge:
            self.image_height -= 2 * self.crop_edge
            self.image_width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.image_width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.image_height / (2 * self.fy))

        self.color_paths = []
        self.depth_paths = []
        self.segmentation_paths = [] ##

        # self.include_segmentation_data = False
        # if "segmentation" in config:
        #     print("SEGMENTATION")
        #     if config["segmentation"]:
        #         self.include_segmentation_data = True
        #         self.segmentation_paths = []

    def __len__(self) -> int:
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)
    

class IsaacDataset(BaseDataset):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # load rgb, depth, and image segmentation images
        self.color_paths = sorted(
            list((self.dataset_path / "rgb").glob("*.png"))
        )
        self.depth_paths = sorted(
            list((self.dataset_path / "depth").glob("*.npy"))
        )
        self.segmentation_paths = sorted(
            list((self.dataset_path / "instance_segmentation").glob("*.pkl"))
        )
        self.bbox_paths = sorted(
            list((self.dataset_path / "bboxes").glob("*.npy"))
        )
        
        # load poses
        self.poses = list(np.load(self.dataset_path / "camera_to_world_transforms.npy"))
        self.poses = [pose.reshape(4, 4).astype(np.float32) for pose in self.poses]

        # transform pose to meet convention: x axis right, y axis down, z axis forward
        for i in range(len(self.poses)):
            self.poses[i][:3, 1] *= -1
            self.poses[i][:3, 2] *= -1
        print(f"Loaded {len(self.color_paths)} frames.")

    def __len__(self) -> int:
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)
    
    def get_annotations(self, idx: int) -> dict:
        segmentation_data = load_pkl(self.segmentation_paths[idx])
        return segmentation_data["annotations"]
    
    def get_bboxes(self, idx: int) -> np.ndarray:
        if self.downsample:
            return np.load(self.bbox_paths[idx]) / 2
        return np.load(self.bbox_paths[idx])
    
    def get_camera_pose(self, idx: int) -> np.ndarray:
        return self.poses[idx]
    
    def get_color_image(self, idx: int) -> np.ndarray:
        color_data = cv2.imread(str(self.color_paths[idx]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        if self.downsample:
            color_data = cv2.resize(color_data, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)

        return color_data
    
    def get_depth_image(self, idx: int) -> np.ndarray:
        depth_data = np.load(str(self.depth_paths[idx]))
        depth_data = depth_data / self.depth_scale
        depth_data[depth_data == 0] = np.nan
        depth_data[depth_data == np.inf] = np.nan

        if self.downsample:
            depth_data = cv2.resize(depth_data, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

        return depth_data
    
    def __getitem__(self, idx: int) -> tuple:
        color_data = cv2.imread(str(self.color_paths[idx]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        depth_data = np.load(str(self.depth_paths[idx]))
        depth_data = depth_data / self.depth_scale
        depth_data[depth_data == 0] = np.nan
        depth_data[depth_data == np.inf] = np.nan

        segmentation_data = load_pkl(self.segmentation_paths[idx])
        segmentation_image = segmentation_data["segmented_image"]
        segmentation_annotation = segmentation_data["annotations"]

        if self.downsample:
            color_data = cv2.resize(color_data, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
            depth_data = cv2.resize(depth_data, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
            segmentation_image = cv2.resize(segmentation_image, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
            
            
        return idx, self.poses[idx], color_data, depth_data, segmentation_image, segmentation_annotation
        
    
class ReplicaDataset(BaseDataset):
    def __init__(self, config: dict, mode: str = "train") -> None:
        super().__init__(config)
        # load rgb and depth images
        self.color_paths = sorted(
            list((self.dataset_path / "results").glob("frame*.jpg"))
        )
        self.depth_paths = sorted(
            list((self.dataset_path / "results").glob("depth*.png"))
        )
        
        # load poses
        self.poses = []
        poses_path = self.dataset_path / "traj.txt"
        with open(poses_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w.astype(np.float32))
        ###
        self.mode = mode
        if mode == "train":
            self.color_paths = self.color_paths[::20]
            self.depth_paths = self.depth_paths[::20]
            self.poses = self.poses[::20]
        elif mode == "val":
            self.color_paths = self.color_paths[10::20]
            self.depth_paths = self.depth_paths[10::20]
            self.poses = self.poses[10::20]
        ###
        print(f"Loaded {len(self.color_paths)} frames.")

    def __len__(self) -> int:
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)

    def __getitem__(self, idx: int) -> tuple:
        color_data = cv2.imread(str(self.color_paths[idx]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(str(self.depth_paths[idx]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return idx, color_data, depth_data, self.poses[idx]


class FlightRoomDataset(BaseDataset):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # load rgb, depth, image segmentation, and depth paths
        self.color_paths = sorted(
            list((self.dataset_path / "rgb").glob("*.png"))
        )
        self.depth_paths = sorted(
            list((self.dataset_path / "depth").glob("*.npy"))
        )
        self.segmentation_paths = sorted(
            list((self.dataset_path / "instance_segmentation").glob("*.pkl"))
        )
        self.bbox_paths = sorted(
            list((self.dataset_path / "bboxes").glob("*.npy"))
        )
        
        # load poses
        self._load_poses()
        
        print(f"Loaded {len(self.color_paths)} frames.")

    def _load_poses(self):
        self.pose_paths = sorted(
            list((self.dataset_path / "pose").glob("*.npy"))
        )

        R_fmu_to_world = np.array([
            [1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0, -1]
        ])
        R_camera_to_quadrotor = np.array([
            [0, 0, 1],  # x -> y
            [1, 0, 0], # y -> -z
            [0, 1, 0]   # z -> x
        ])

        self.poses = []
        for path in self.pose_paths:
            pose_data = np.load(path)

            # extract and transform position
            position = R_fmu_to_world @ pose_data[:3]

            # extract and transform quaternion
            quaternion = pose_data[3:]
            R_quadrotor_to_fmu = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()
            rotation = R_fmu_to_world @ R_quadrotor_to_fmu @ R_camera_to_quadrotor

            # assemble 4 x 4 transformation matrix and append to poses
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = position
            self.poses.append(pose)
        
    def __len__(self) -> int:
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)
    
    def get_annotations(self, idx: int) -> dict:
        segmentation_data = load_pkl(self.segmentation_paths[idx])
        return segmentation_data["annotations"]
    
    def get_bboxes(self, idx: int) -> np.ndarray:
        if self.downsample:
            return np.load(self.bbox_paths[idx]) / 2
        return np.load(self.bbox_paths[idx])
    
    def get_camera_pose(self, idx: int) -> np.ndarray:
        return self.poses[idx]
    
    def get_color_image(self, idx: int) -> np.ndarray:
        color_data = cv2.imread(str(self.color_paths[idx]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        if self.downsample:
            color_data = cv2.resize(color_data, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)

        return color_data
    
    def get_depth_image(self, idx: int) -> np.ndarray:
        depth_data = np.load(str(self.depth_paths[idx]))
        depth_data = depth_data / self.depth_scale
        depth_data[depth_data == 0] = np.nan
        depth_data[depth_data == np.inf] = np.nan

        if self.downsample:
            depth_data = cv2.resize(depth_data, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)

        return depth_data
    
    def __getitem__(self, idx: int) -> tuple:
        color_data = cv2.imread(str(self.color_paths[idx]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        depth_data = np.load(str(self.depth_paths[idx]))
        depth_data = depth_data / self.depth_scale
        depth_data[depth_data == 0] = np.nan
        depth_data[depth_data == np.inf] = np.nan

        segmentation_data = load_pkl(self.segmentation_paths[idx])
        segmentation_image = segmentation_data["segmented_image"]
        segmentation_annotation = segmentation_data["annotations"]

        if self.downsample:
            color_data = cv2.resize(color_data, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
            depth_data = cv2.resize(depth_data, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
            segmentation_image = cv2.resize(segmentation_image, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
            
            
        return idx, self.poses[idx], color_data, depth_data, segmentation_image, segmentation_annotation


class TUM_RGBD(BaseDataset):
    def __init__(self, config: dict, mode: str = "train"):
        super().__init__(config)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.dataset_path, frame_rate=10)
        print(f"Loaded {len(self.color_paths)} frames.")
        
        ###
        # self.mode = mode
        # if mode == "train":
        #     self.color_paths = self.color_paths[::6]
        #     self.depth_paths = self.depth_paths[::6]
        #     self.poses = self.poses[::6]
        # elif mode == "val":
        #     self.color_paths = self.color_paths[3::6]
        #     self.depth_paths = self.depth_paths[3::6]
        #     self.poses = self.poses[3::6]
        ###

    def __len__(self) -> int:
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        return np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))
            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))
        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indices = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indices[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indices += [i]

        images, poses, depths = [], [], []
        inv_pose = None
        for ix in indices:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            poses += [c2w.astype(np.float32)]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.K, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, self.poses[index]
    

def get_dataset(dataset_name: str) -> BaseDataset:
    if dataset_name == "replica":
        return ReplicaDataset
    elif dataset_name == "tum_rgbd":
        return TUM_RGBD
    elif dataset_name == "isaac":
        return IsaacDataset
    elif dataset_name == "flightroom":
        return FlightRoomDataset
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    

if __name__ == "__main__":
    # config = {
    #     "dataset_path": "../../data/Replica-SLAM/office0",
    #     "H": 680,
    #     "W": 1200,
    #     "fx": 600.0,
    #     "fy": 600.0,
    #     "cx": 599.5,
    #     "cy": 339.5,
    #     "depth_scale": 6553.5,
    # }

    # dataset = ReplicaDataset(config)
    # breakpoint()

    config = {
        "dataset_path": "../../data/TUM_RGBD-SLAM/rgbd_dataset_freiburg1_desk",
        "H": 480,
        "W": 640,
        "fx": 517.3,
        "fy": 516.5,
        "cx": 318.6,
        "cy": 255.3,
        "crop_edge": 50,
        "distortion": [0.2624, -0.9531, -0.0054, 0.0026, 1.1633],
        "depth_scale": 5000.0,
    }

    dataset = TUM_RGBD(config)
    breakpoint()