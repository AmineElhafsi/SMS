from typing import Optional

import numpy as np
import omni.isaac.core.utils.prims as prims_utils

from omni.isaac.core.objects import VisualSphere

def check_object_overlap(aabb1, aabb2, margin=0.0):
    # unpack the AABBs
    x_min1, y_min1, z_min1, x_max1, y_max1, z_max1 = aabb1
    x_min2, y_min2, z_min2, x_max2, y_max2, z_max2 = aabb2

    # apply margin
    x_min1, y_min1, z_min1 = x_min1 - margin, y_min1 - margin, z_min1 - margin
    x_max1, y_max1, z_max1 = x_max1 + margin, y_max1 + margin, z_max1 + margin
    
    # check overlap along the X-axis
    overlap_x = (x_max1 >= x_min2) and (x_max2 >= x_min1)
    
    # check overlap along the Y-axis
    overlap_y = (y_max1 >= y_min2) and (y_max2 >= y_min1)
    
    # check overlap along the Z-axis
    overlap_z = (z_max1 >= z_min2) and (z_max2 >= z_min1)
    
    # AABBs overlap if they overlap along all three axes
    return overlap_x and overlap_y and overlap_z

def debug_marker(position: np.ndarray, color: Optional[np.ndarray] = None, size: float = 0.1):
    # check if object under the same prim path already exists
    instance_id = 0
    prim_path = f"/World/Debug/Marker_{instance_id}"
    while prims_utils.is_prim_path_valid(prim_path):
        # increment instance ID and update prim path
        instance_id += 1
        prim_path = f"/World/Debug/Marker_{instance_id}"
    prim = VisualSphere(
        prim_path=prim_path, 
        position=position,
        color=color,
        radius=size,
    )