import numpy as np
import omni

import omni.isaac.core.utils.prims as prims_utils

from omni.isaac.core import World
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.prims import GeometryPrim, RigidPrim, XFormPrim
from pxr import Usd

from materials.utils import apply_physics_material_to_prim


class FrankaFR3:
    def __init__(self, config):
        # config
        self.config = config
        
        # robot_config = self.config["isaac"]["robot"]

        # set up world
        self.world = World()

        # load robot from URDF
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.create_physics_scene = True
        import_config.fix_base = True
        import_config.make_default_prim = True
        import_config.merge_fixed_joints = False
        import_config.import_inertia_tensor = True
        import_config.set_convex_decomp(True)
        result, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=config["urdf_path"],
            import_config=import_config,
        )

        self.joint_names = self.config["joint_names"]
        self.default_joint_positions = np.array(self.config["default_joint_positions"])

        # wrap the prim as a robot (articulation)
        self.robot = Robot(
            prim_path=prim_path, 
            name="franka_panda",
            position=np.array(config["position"]),
        )
        self.world.render()
        
        # prepare Xform and Geometry prims for future robot property configuration
        self.xform_prim = XFormPrim(prim_path=prim_path)

        # set physics material for robot hand (density unset as mass is set in URDF)
        robot_prim = prims_utils.get_prim_at_path(prim_path)
        apply_physics_material_to_prim(robot_prim, "aluminum", set_density=False)            

        # self.world.render()
        
        self.world.step()
        self.robot.initialize()
        self.reset()

    def reset(self):
        # try:
        #     self.robot.initialize()
        # except Exception as e:
        #     print("Failed to initialize robot:", e)

        # set initial state
        controlled_joint_indices = [self.robot.dof_names.index(joint_name) for joint_name in self.joint_names]
        # default_joint_positions = np.array(self.config["default_joint_positions"])
        self.robot.set_joint_positions(self.default_joint_positions, joint_indices=controlled_joint_indices)

        # close gripper
        self.robot.set_joint_positions([0.0, 0.0], joint_indices=[7, 8])
        self.world.step() 

        # set base position (because omni.isaac.core.robots.Robot constructor is not setting the base position correctly)
        self.xform_prim.set_world_pose(position=np.array(self.config["position"]))
        self.world.render()