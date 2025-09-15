from typing import Dict, List

import numpy as np

from scipy.optimize import minimize

### Friction ###
# References:
# https://www.engineeringtoolbox.com/friction-coefficients-d_778.html
STATIC_FRICTION_PAIRS = {
    ("aluminum", "aluminum"): 1.05,
    ("aluminum", "steel"): 0.61,
    ("aluminum", "wood"): 0.6,
    ("brass", "brass"): 0.425,
    ("brass", "steel"): 0.51,
    ("cardboard", "cardboard"): 0.5,
    ("cardboard", "plastic"): 0.4,
    ("cardboard", "wood"): 0.4,
    ("polished_ceramic", "polished_ceramic"): 0.4,
    ("rough_ceramic", "rough_ceramic"): 0.6,
    ("copper", "copper"): 1.6,
    ("copper", "steel"): 0.53,
    ("plastic", "plastic"): 0.3,
    ("plastic", "wood"): 0.4,
    ("resin", "resin"): 0.055,
    ("rubber", "rubber"): 1.16,
    ("rubber", "cardboard"): 0.65,
    ("rubber", "stone"): 1.0,
    ("steel", "steel"): 0.65,
    ("steel", "stone"): 0.4,
    ("stone", "stone"): 0.6,
    ("stone", "wood"): 0.6,
    ("wood", "wood"): 0.5,
}

KINETIC_FRICTION_PAIRS = {
    ("aluminum", "aluminum"): 0.4,
    ("aluminum", "steel"): 0.47,
    ("aluminum", "wood"): 0.35,
    ("brass", "brass"): 0.35,
    ("brass", "steel"): 0.44,
    ("cardboard", "cardboard"): 0.35,
    ("cardboard", "plastic"): 0.25,
    ("cardboard", "wood"): 0.25,
    ("polished_ceramic", "polished_ceramic"): 0.2,
    ("rough_ceramic", "rough_ceramic"): 0.5,
    ("copper", "copper"): 1.0,
    ("copper", "steel"): 0.36,
    ("plastic", "plastic"): 0.2,
    ("plastic", "wood"): 0.3,
    ("resin", "resin"): 0.05,
    ("rubber", "rubber"): 1.0,
    ("rubber", "stone"): 0.75,
    ("rubber", "cardboard"): 0.5,
    ("steel", "steel"): 0.42,
    ("steel", "stone"): 0.35,
    ("stone", "stone"): 0.5,
    ("stone", "wood"): 0.5,
    ("wood", "wood"): 0.4,
}

def estimate_friction(material_pairs: Dict):
    """
    Estimate per-object friction values given a table of material-pair friction values.
    Uses least-squares optimization with optional regularization for missing data.
    
    Args:
        material_pairs (dict): Dictionary with keys as (material1, material2) tuples and values as friction coefficients.
        regularization (float): Weight for the regularization term to prevent extreme values.
    
    Returns:
        dict: Estimated per-object friction values.
    """
    # extract unique materials
    materials = list(set(mat for pair in material_pairs.keys() for mat in pair))
    num_materials = len(materials)
    
    # objective function to minimize the error in friction estimation
    def friction_error(mu_values):
        mu_dict = {materials[i]: mu_values[i] for i in range(num_materials)}
        error = 0.0
        
        for (mat1, mat2), mu_eff in material_pairs.items():
            predicted_mu_eff = mu_dict[mat1] * mu_dict[mat2]
            error += (predicted_mu_eff - mu_eff) ** 2  # Squared error
                
        return error
    
    # initial guess and bounds friction values
    initial_guess = np.ones(num_materials) * 0.5    
    bounds = [(0.01, 5.0)] * num_materials
    
    # solve
    result = minimize(friction_error, initial_guess, bounds=bounds)
    
    # return result
    optimized_friction = {materials[i]: result.x[i] for i in range(num_materials)}
    
    return optimized_friction

STATIC_FRICTION_COEFFICIENTS = estimate_friction(STATIC_FRICTION_PAIRS)
KINETIC_FRICTION_COEFFICIENTS = estimate_friction(KINETIC_FRICTION_PAIRS)

### Restitution ###
# References:
# https://cdnsciencepub.com/doi/pdf/10.1139/cgj-2018-0712
# https://www.researchgate.net/publication/261215109_NOVEL_TECHNICHES_FOR_EXPERIMENTAL_DETERMINATION_OF_THE_RESTITUTION_COEFFICIENT_BY_MEANS_OF_ACOUSTIC_SIGNAL_ANALYSIS
# https://repository.rice.edu/server/api/core/bitstreams/6748ea32-56bf-43a4-88cd-08b757647052/content

RESTITUTION_COEFFICIENTS = {
    "aluminum": 0.75,
    "brass": 0.54,
    "cardboard": 0.3,
    "polished_ceramic": 0.65,
    "rough_ceramic": 0.55,
    "copper": 0.6,
    "plastic": 0.8,
    "resin": 0.9,
    "rubber": 0.38,
    "steel": 0.69,
    "stone": 0.5,
    "wood": 0.6,
}

### Density (kg/m^3) ###
# References:
# https://www.engineeringtoolbox.com/density-gravity-t_64.html
# Miscellaneous reference sheets
DENSITIES = {
    "aluminum": 2700,
    "brass": 8550,
    "cardboard": 600,
    "polished_ceramic": 2450,
    "rough_ceramic": 1826,
    "copper": 8940,
    "plastic": 1350,
    "resin": 1780,
    "rubber": 1200,
    "steel": 7850,
    "stone": 2550,
    "wood": 800,
}

### Special Cases ###
STATIC_FRICTION_COEFFICIENTS["marble"] = STATIC_FRICTION_COEFFICIENTS["polished_ceramic"]
KINETIC_FRICTION_COEFFICIENTS["marble"] = KINETIC_FRICTION_COEFFICIENTS["polished_ceramic"]
RESTITUTION_COEFFICIENTS["marble"] = 0.92
DENSITIES["marble"] = 2700

STATIC_FRICTION_COEFFICIENTS["pumpkin"] = STATIC_FRICTION_COEFFICIENTS["wood"]
KINETIC_FRICTION_COEFFICIENTS["pumpkin"] = KINETIC_FRICTION_COEFFICIENTS["wood"]
RESTITUTION_COEFFICIENTS["pumpkin"] = 0.25
DENSITIES["pumpkin"] = 330

STATIC_FRICTION_COEFFICIENTS["tennis_ball"] = STATIC_FRICTION_COEFFICIENTS["rubber"]
KINETIC_FRICTION_COEFFICIENTS["tennis_ball"] = KINETIC_FRICTION_COEFFICIENTS["rubber"]
RESTITUTION_COEFFICIENTS["tennis_ball"] = 0.73
DENSITIES["tennis_ball"] = 382

STATIC_FRICTION_COEFFICIENTS["soccer_ball"] = STATIC_FRICTION_COEFFICIENTS["plastic"]
KINETIC_FRICTION_COEFFICIENTS["soccer_ball"] = KINETIC_FRICTION_COEFFICIENTS["plastic"]
RESTITUTION_COEFFICIENTS["soccer_ball"] = 0.8
DENSITIES["soccer_ball"] = 80

### Aggregate data ###

def ensure_same_keys(dicts: List[Dict]):
    # get the set of all keys from all dictionaries
    all_keys = set().union(*dicts)
    
    # ensure each dictionary has all keys
    for d in dicts:
        for key in all_keys:
            if key not in d:
                raise KeyError(f"Key '{key}' not found in dictionary.")

# ensure all data sheets have the same keys
ensure_same_keys(
    [
        STATIC_FRICTION_COEFFICIENTS, 
        KINETIC_FRICTION_COEFFICIENTS, 
        RESTITUTION_COEFFICIENTS, 
        DENSITIES
    ]
)

# get list of materials
MATERIALS_LIST = list(DENSITIES.keys())