from typing import Optional

import omni.isaac.core.utils.prims as prims_utils

from omni.isaac.core import World
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import GeometryPrim, RigidPrim
from pxr import Usd, PhysxSchema

from src.isaac_sim.materials.data_sheet import (
    STATIC_FRICTION_COEFFICIENTS,
    KINETIC_FRICTION_COEFFICIENTS,
    RESTITUTION_COEFFICIENTS,
    DENSITIES,
    MATERIALS_LIST
)

# def add_materials_to_stage():
#     """
#     Add physics materials from the data sheet to the stage.
#     """
#     # refresh world
#     world = World()
#     world.render()

#     for material in MATERIALS_LIST:
#         print("material: ", material)
#         mu_s = STATIC_FRICTION_COEFFICIENTS[material]
#         mu_k = KINETIC_FRICTION_COEFFICIENTS[material]
#         restitution = RESTITUTION_COEFFICIENTS[material]

#         material_prim_path = "/Materials/" + material
#         if prims_utils.is_prim_path_valid(material_prim_path):
#             print(f"Material '{material}' already exists in stage.")
#             continue

#         physics_material = PhysicsMaterial(
#             material_prim_path,
#             static_friction=mu_s,
#             dynamic_friction=mu_k,
#             restitution=restitution,
#         )

#         # set friction combine mode to "multiply" and restitution combine mode to "min"
#         PhysxSchema.PhysxMaterialAPI.Apply(physics_material.prim)
#         physics_material.prim.GetAttribute("physxMaterial:frictionCombineMode").Set("multiply")
#         physics_material.prim.GetAttribute("physxMaterial:restitutionCombineMode").Set("min")

#     # refresh world
#     world.render()

def apply_physics_material_to_prim(
    prim: Usd.Prim,
    material: str, 
    set_density: bool,
):
    """
    Apply a physics material to a prim from the material datashet. If the material does not 
    exist in the stage, create it.

    Args:
        prim: The prim to apply the physics material to.
        material: The name of the material to apply.
        set_density: Whether to set the density of the prim. If True, the density is set to the 
            value in the data sheet.
    """
    assert material in MATERIALS_LIST, f"Material '{material}' not found in data sheet."

    # refresh world
    world = World()
    # world.render()

    # set density (needs to be set separately from physics material)
    if set_density:
        density = DENSITIES[material]
        rigid_prim = RigidPrim(prim_path=prim.GetPath().pathString)
        rigid_prim.set_density(density)

    # create physics material    
    material_prim_path = "/Materials/" + material
    mu_s = STATIC_FRICTION_COEFFICIENTS[material]
    mu_k = KINETIC_FRICTION_COEFFICIENTS[material]
    restitution = RESTITUTION_COEFFICIENTS[material]

    physics_material = PhysicsMaterial(
        material_prim_path,
        static_friction=mu_s,
        dynamic_friction=mu_k,
        restitution=restitution,
    )

    # set friction combine mode to "multiply" and restitution combine mode to "min"
    PhysxSchema.PhysxMaterialAPI.Apply(physics_material.prim)
    physics_material.prim.GetAttribute("physxMaterial:frictionCombineMode").Set("multiply")
    physics_material.prim.GetAttribute("physxMaterial:restitutionCombineMode").Set("min")

    # wrap prim in GeometryPrim and apply material
    geometry_prim = GeometryPrim(prim_path=prim.GetPath().pathString)
    # world.render()
    geometry_prim.apply_physics_material(
        physics_material
    )

    # refresh world
    world.render()