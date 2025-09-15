import omni.isaac.core.utils.prims as prims_utils

def add_physics_materials_to_stage():
    physics_materials_usd_paths = {
        "asphalt": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/asphalt_mat.usda",
        "cardboard": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/cardboard_mat.usda",
        "concrete": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/concrete_mat.usda",
        "fabric": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/fabric_mat.usda",
        "glass": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/glass_mat.usda",
        "leaf": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/leaf_mat.usda",
        "leather": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/leather_mat.usda",
        "metal": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/metal_mat.usda",
        "organic": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/organic_mat.usda",
        "plastic": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/plastic_mat.usda",
        "rubber": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/rubber_mat.usda",
        "stone": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/stone_mat.usda",
        "vinyl": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/vinyl_mat.usda",
        "wood": "omniverse://localhost/NVIDIA/Assets/DigitalTwin/Materials/physics_materials/wood_mat.usda",
    }

    if not prims_utils.is_prim_path_valid("/World/PhysicsMaterials"):
        material_prim = prims_utils.create_prim(
            prim_path="/World/PhysicsMaterials",
            prim_type="Scope",
        )
        
    for material_name, material_usd_path in physics_materials_usd_paths.items():
        material_prim = prims_utils.create_prim(
            prim_path=f"/World/PhysicsMaterials/{material_name}",
            prim_type="Material",
            usd_path=material_usd_path,
        )
