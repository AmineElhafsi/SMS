import omni.usd

from omni.isaac.core import World
from omni.physx import  acquire_physx_interface
from pxr import Usd, UsdGeom

from scenes.entities import SceneProp


# stage = omni.usd.get_context().get_stage()

class BaseScene:
    def __init__(self, config):
        self.config = config
        
        # simulation parameters
        self.dt = self.config["isaac"]["dt"]
        
        # set up world
        self.world = World(physics_dt=self.dt, stage_units_in_meters=1.0)
        self.world.render()
        self.world.clear()
        self.world.reset()
        self.world.render()

        # get physx interface
        self.physx_interface = acquire_physx_interface() 

        # add robot
        self.robot = None
        self.controller = None

        # clear any potentially existing content
        self.clear()
        self.reset()

        # set up scene
        self.setup_base_scene()
        self.scene_props = {}

    def reset(self):
        self.world.reset()

        self.physx_interface.reset_simulation()
        self.physx_interface.update_transformations(False, True)

        if self.robot is not None:
            self.robot.reset()

        if self.controller is not None:
            self.controller.reset()

    def clear(self):
        self.world.clear()
        self.scene_props = {}

    def step(self):
        self.physx_interface.update_simulation(self.dt, self.world.current_time)
        self.physx_interface.update_transformations(False, True)
        self.world.render()

    def add_scene_prop(self, name: str, scene_prop: SceneProp):
        if name in self.scene_props:
            raise ValueError(f"Scene prop with name {name} already exists")
        self.scene_props[name] = scene_prop

    def setup_base_scene(self):
        raise NotImplementedError
    
    def add_robot(self):
        raise NotImplementedError
    
    def add_controller(self):
        raise NotImplementedError
    
    def generate_scenario(self):
        raise NotImplementedError
    