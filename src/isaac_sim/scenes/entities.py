from typing import Optional, Union

import omni
import omni.isaac.core.utils.bounds as bounds_utils
import omni.isaac.core.utils.prims as prims_utils
import numpy as np

from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import GeometryPrim, RigidPrim, XFormPrim
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.dynamic_control import _dynamic_control as dc
from omni.physx.scripts import utils as physx_utils
from pxr import Usd, UsdPhysics

from materials.utils import apply_physics_material_to_prim


class SceneProp:
    def __init__(
        self,
        usd_path: str, 
        group: Optional[str] = "Miscellaneous",
        name: Optional[str] = None,
        position: np.array = np.array([0., 0., 0.]),
        orientation: np.array = np.array([1., 0., 0., 0.]),
        scale_factor: Union[float, np.array] = 1.0,
        add_semantics: bool = True,
        semantic_label: Optional[str] = None,
        enable_physics: bool = True,
        physics_material_name: Optional[str] = None
    ):
        # store info
        self.usd_path = usd_path
        self.group = group
        self.name = name
        self.position = position
        self.orientation = orientation
        self.scale_factor = scale_factor
        self.setting_add_semantics = add_semantics
        self.semantic_label = semantic_label
        self.setting_enable_physics = enable_physics
        self.physics_material_name = physics_material_name

        # initializations
        if name is None:
            name = usd_path.split("/")[-1].split(".")[0]
            instance_id = 0
            prim_path = f"/World/{group}/{name}_instance_{instance_id}"
        else:
            instance_id = 0
            prim_path = f"/World/{group}/{name}_instance_{instance_id}"
        self.name = name

        # check if prop under the same prim path already exists
        while prims_utils.is_prim_path_valid(prim_path):
            # increment instance ID and update prim path
            instance_id += 1
            prim_path = f"/World/{group}/{name}_instance_{instance_id}"
        self.prim_path = prim_path
        
        # create prop prim
        self.prim = prims_utils.create_prim(
            prim_path=prim_path,
            prim_type="Xform",
            usd_path=usd_path,
            position=position,
            orientation=orientation,
            scale=np.array([0.01, 0.01, 0.01]) * scale_factor
        )
        self.xform_prim = XFormPrim(
            prim_path,
            scale=np.array([0.01, 0.01, 0.01]) * scale_factor
        )

        # add semantics
        if add_semantics:
            if semantic_label is None:
                semantic_label = name
            self.add_semantic_class(semantic_label=semantic_label)

        # enable physics
        if enable_physics:
            self.enable_physics()

        # apply physics material
        if physics_material_name is not None:
            self.apply_physics_material(physics_material_name)

        # misc
        if "Camera" in self.prim.GetPath().pathString:
            strap_prim = prims_utils.get_prim_at_path(self.prim.GetPath().pathString + "/Camera_01_strap")
            strap_prim.SetActive(False)

    def add_semantic_class(self, semantic_label: str):
        for prim in Usd.PrimRange(self.prim):
            if prim.GetTypeName() == "Mesh":
                add_update_semantics(prim, semantic_label=semantic_label, type_label="class")

    def apply_physics_material(self, material_name: str):
        apply_physics_material_to_prim(self.prim, material_name, set_density=True)


        # physics_material = PhysicsMaterial(
        #     f"/World/PhysicsMaterials/{material_name}"
        # )

        # geometry_prim = GeometryPrim(prim_path=self.prim.GetPrimPath().pathString)
        # geometry_prim.apply_physics_material(physics_material)

        # # apply mass/density separately because of isaac sim api design decisions
        # material_density = physics_material.prim.GetAttribute("physics:density").Get()
        # self.rigid_prim = RigidPrim(prim_path=self.prim.GetPrimPath().pathString, density=1000)

    def enable_physics(self):
        if "ball" in self.name.lower():
            physx_utils.setRigidBody(self.prim, "boundingSphere", kinematic=False)
        else:
            physx_utils.setRigidBody(self.prim, "convexDecomposition", kinematic=False)
        # for prim in Usd.PrimRange(self.prim):
        #     if prim.GetTypeName() == "Mesh":
        #         prim.GetAttribute(
        #             "physxConvexDecompositionCollision:shrinkWrap"
        #         ).Set(True)
        #         prim.GetAttribute(
        #             "physxConvexDecompositionCollision:errorPercentage"
        #         ).Set(0.015)

    def get_aabb(self):
        cache = bounds_utils.create_bbox_cache()
        (min_x, min_y, min_z, max_x, max_y, max_z) = \
            bounds_utils.compute_aabb(cache, prim_path=self.prim_path, include_children=True)
        return min_x, min_y, min_z, max_x, max_y, max_z

    def get_aabb_dimensions(self):
        (min_x, min_y, min_z, max_x, max_y, max_z) = self.get_aabb()
        dx, dy, dz = max_x - min_x, max_y - min_y, max_z - min_z
        return dx, dy, dz
    
    def get_mass(self):
        _dc = dc.acquire_dynamic_control_interface()
        _selected_handle = _dc.get_rigid_body(self.prim.GetPrimPath().pathString)
        rigid_body_properties = _dc.get_rigid_body_properties(_selected_handle)
        return round(rigid_body_properties.mass, 3)

    def get_world_com(self):
        _dc = dc.acquire_dynamic_control_interface()
        _selected_handle = _dc.get_rigid_body(self.prim.GetPrimPath().pathString)
        rigid_body_properties = _dc.get_rigid_body_properties(_selected_handle)
        local_com = np.array(
            [
                rigid_body_properties.cMassLocalPose.x,
                rigid_body_properties.cMassLocalPose.y,
                rigid_body_properties.cMassLocalPose.z
            ]
        )
        world_position = self.xform_prim.get_world_pose()[0]
        world_com = world_position + local_com

        return world_com
    
    def get_world_position(self):
        return self.xform_prim.get_world_pose()[0]
    
    def get_orientation(self):
        return self.xform_prim.get_world_pose()[1]
    
    def remove_from_stage(self):
        prims_utils.delete_prim(self.prim_path)
    
    def set_default_state(self, position: np.array = None, orientation: np.array = None):
        # rigid_prim = RigidPrim(prim_path=self.prim.GetPrimPath().pathString, density=1000)
        # rigid_prim.set_default_state(position=position, orientation=orientation)
        # rigid_prim.post_reset()
        print("default state before: ", self.xform_prim.get_default_state().position)
        self.xform_prim.set_default_state(position=position, orientation=orientation)
        self.xform_prim.post_reset()
        print("default state after: ", self.xform_prim.get_default_state().position)

    def set_linear_velocity(self, linear_velocity: np.array):
        self.rigid_prim.set_linear_velocity(linear_velocity)

    def set_world_pose(self, position: np.array = None, orientation: np.array = None):
        self.xform_prim.set_world_pose(position=position, orientation=orientation)

    def snapshot(self):
        # assemble snapshot dictionary
        current_position = self.get_world_position()
        current_orientation = self.get_orientation()

        snapshot_dict = {
            "usd_path": self.usd_path,
            "group": self.group,
            "name": self.name,
            "position": current_position.tolist(),
            "orientation": current_orientation.tolist(),
            "scale_factor": self.scale_factor.tolist() if isinstance(self.scale_factor, np.ndarray) else self.scale_factor,
            "add_semantics": self.setting_add_semantics,
            "semantic_label": self.semantic_label,
            "enable_physics": self.setting_enable_physics,
            "physics_material_name": self.physics_material_name
        }

        return snapshot_dict