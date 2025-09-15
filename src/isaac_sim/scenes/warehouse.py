from typing import Dict, List, Optional, Tuple

import omni
import omni.isaac.core.utils.prims as prims_utils
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import GeometryPrim, RigidPrim, XFormPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.types import ArticulationAction
from pxr import UsdPhysics

from materials.utils import apply_physics_material_to_prim
from scenes.base import BaseScene
from scenes.entities import SceneProp
from scenes.props_lists import BILLIARDS_OBSTACLES
from scenes.simple_room_helpers import randomly_place_billiard_balls, randomly_place_obstacles
from scenes.utils import debug_marker
from src.controllers.manipulator import BilliardsManipulatorController
from src.controllers.utils import generate_strike_reference_motion

from robots.manipulators import FrankaFR3


class WarehouseScene(BaseScene):
    def __init__(self, config):
        super().__init__(config)

    # def __init__(self, world: World):
    #     self.world = world
    #     self.clear()

    #     # set up scene
    #     self.setup_base_scene()
    #     self.scene_props = {}

    def clear(self):
        self.world.clear()

    def reset(self):
        self.world.reset()

    def update(self):
        self.world.render()
        self.world.reset()
        self.world.render()

    def setup_base_scene(self):
        print("Setting up empty scene...")
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise FileNotFoundError("Could not find assets root path")
        usd_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        print("assets_root_path: ", assets_root_path)
        # omniverse://localhost/NVIDIA/Assets/Isaac/4.1/Isaac/Environments/Simple_Room/simple_room.usd
        warehouse_prim = prims_utils.create_prim(
            prim_path="/World/Warehouse",
            prim_type="Xform",
            usd_path=usd_path,
        )
        self.update()
        print("Scene setup complete.")