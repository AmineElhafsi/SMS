import numpy as np
import omni.isaac.core.utils.numpy as numpy_utils
import omni.isaac.core.utils.prims as prims_utils

from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera
from scipy.spatial.transform import Rotation

def quaternion_to_rotation_matrix(q: np.ndarray):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    
    # Compute the rotation matrix
    R = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                  [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                  [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])
    
    return R


class OrbbecGemini2:
    def __init__(self, world, prim_path: str = "/World/RGBD_Camera"):
        self.prim_path = prim_path

        # create object prim
        self.prim = prims_utils.create_prim(
            usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Sensors/Orbbec/Gemini 2/orbbec_gemini2_V1.0.usd",
            prim_path=prim_path
        )
        self.camera = Camera(
            prim_path=prim_path + "/Orbbec_Gemini2/camera_ir_left/camera_left/Stream_depth",
            dt=-1,
            resolution=(1280, 800)
        )
        self.xform_prim = XFormPrim(
            self.prim_path        
        )

        # turn off flash (hack to get rid of white artifact at camera target)
        light_prim = prims_utils.get_prim_at_path("/World/RGBD_Camera/Orbbec_Gemini2/camera_ldm/camera_ldm/RectLight")
        prims_utils.set_prim_visibility(light_prim, False)

        # initialize and set up depth capture
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()
        self.camera.add_instance_segmentation_to_frame()

        # set world object
        self.world = world
        self.world.render()

    def get_camera_to_world_transform(self):
        self.world.render()
        camera_position, q_c2w = self.camera.get_world_pose(camera_axes="usd")
        R_c2w = quaternion_to_rotation_matrix(q_c2w)
        T = np.eye(4)
        T[:3, :3] = R_c2w
        T[:3, 3] = camera_position
        return T

    def get_data(self, render_steps: int = 8):
        for _ in range(render_steps):
            self.world.render()
        return self.camera.get_current_frame()

    def point_at(self, target_position: np.ndarray):
        # calculate direction from camera to target
        camera_position, _ = self.xform_prim.get_world_pose()
        direction = target_position - camera_position
        direction = direction / np.linalg.norm(direction)

        # set upward direction
        up_direction = np.array([0., 0., 1.])
        # avoid degenerate case where direction is nearly parallel to up_direction
        if np.abs(np.dot(direction, up_direction)) > 0.99:
            up_direction = np.array([1., 0., 0.])

        # calculate the right vector
        right = np.cross(up_direction, direction)
        right = right / np.linalg.norm(right)

        # calculate the new up vector
        up_direction = np.cross(direction, right)

        # construct rotation matrix
        rotation_matrix = np.column_stack((right, up_direction, direction))
        rotation = Rotation.from_matrix(rotation_matrix).as_euler('xyz')
        cam_orientation = numpy_utils.euler_angles_to_quats(rotation)
        self.xform_prim.set_world_pose(orientation=cam_orientation)

    def set_world_pose(self, position: np.ndarray = None, orientation: np.ndarray = None):
        self.xform_prim.set_world_pose(position=position, orientation=orientation)