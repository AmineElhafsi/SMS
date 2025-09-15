from typing import Dict, List, Optional, Tuple

import numpy as np

from omni.isaac.core.utils.rotations import euler_angles_to_quat

from scenes.entities import SceneProp
from scenes.utils import check_object_overlap

def randomly_place_billiard_balls(
    ball_numbers: List[int],
    max_balls: int,
    r_range: Tuple[float, float],
    theta_range: Tuple[float, float],
    ball_buffer_margin: float,
    offset_position: np.ndarray = np.array([0., 0., 0.03]),
    world = None,
    existing_scene_props: Optional[Dict[str, SceneProp]] = None,
    max_placement_attempts: int = 20,
) -> Dict:
    
    billiard_ball_dict = {}

    num_balls = np.random.choice(np.array(range(max_balls))+1)
    print("Number of obstacle balls: ", num_balls)
    selected_ball_numbers = np.random.choice(ball_numbers, num_balls, replace=False)

    for number in selected_ball_numbers:
        candidate_prop_exists = False
        placement_successful = False
        placement_attempt_counter = 0
        while True:
            if placement_attempt_counter >= max_placement_attempts:
                print("Unsuccessful billiard ball placement after", max_placement_attempts, "attempts")
                candidate_prop.remove_from_stage()
                del candidate_prop
                world.render()
                break

            r_ball = np.random.uniform(*r_range)
            theta_ball = np.random.uniform(*theta_range)
            new_ball_position = np.array(
                [r_ball * np.cos(theta_ball), r_ball * np.sin(theta_ball), 0.03 * 1.3]
            ) + offset_position

            if candidate_prop_exists:
                    candidate_prop.set_world_pose(position=new_ball_position)
            else:
                candidate_prop = SceneProp(
                    usd_path=f"file:/home/anonymous/Documents/Research/simulation/assets/props/billiard_balls/BilliardBalls_{number:02}.usd",
                    group="BilliardBalls",
                    position=new_ball_position,
                    scale_factor=1.3,
                    enable_physics=True,
                    physics_material_name="resin"
                )
                candidate_prop_exists = True
            world.render()

            # check for overlap with existing balls
            overlap = False
            for _, prop in existing_scene_props.items():
                overlap = check_object_overlap(candidate_prop.get_aabb(), prop.get_aabb(), margin=ball_buffer_margin)
                if overlap:
                    placement_attempt_counter += 1
                    break

            # if successful placement, add to existing_scene_props and break out of the loop
            if not overlap:
                placement_successful = True
                break

        if placement_successful:
            existing_scene_props.update(
                {"obstacle_billiard_ball_" + str(number): candidate_prop}
            )

def randomly_place_obstacles(
    obstacle_choices: List[Dict],
    max_obstacles: int,
    r_range: Tuple[float, float],
    theta_range: Tuple[float, float],
    prop_buffer_margin: float,
    offset_position: np.ndarray = np.array([0., 0., 0.0]),
    world=None,
    existing_scene_props: Optional[Dict[str, SceneProp]] = None,
    max_placement_attempts: int = 20,
):
    
    num_obstacles = np.random.choice(np.array(range(max_obstacles))+1)
    obstacles = np.random.choice(obstacle_choices, num_obstacles, replace=False)
    print("Number of obstacles: ", num_obstacles)

    for obstacle in obstacles:
        candidate_prop_exists = False
        placement_successful = False
        placement_attempt_counter = 0
        while True:
            if placement_attempt_counter >= max_placement_attempts:
                print(f"Unsuccessful obstacle placement, {obstacle} ,after", max_placement_attempts, "attempts")
                candidate_prop.remove_from_stage()
                del candidate_prop
                world.render()
                break

            r_obstacle = np.random.uniform(*r_range)
            theta_obstacle = np.random.uniform(*theta_range)
            new_obstacle_position = np.array(
                [r_obstacle * np.cos(theta_obstacle), r_obstacle * np.sin(theta_obstacle), 0.0]
            ) + offset_position
            new_obstacle_orientation = euler_angles_to_quat([0., 0., np.random.uniform(0., 2*np.pi)])
            scale_factor = np.random.uniform(*obstacle["scale_range"]) if "scale_range" in obstacle else 1.0

            name = obstacle["name"]
            print(f"Attempting to add obstacle {name} with scale factor of {scale_factor}")

            if candidate_prop_exists:
                candidate_prop.set_world_pose(position=new_obstacle_position, orientation=new_obstacle_orientation)
            else:
                candidate_prop = SceneProp(
                    usd_path=obstacle["usd_path"],
                    group=obstacle["group"],
                    position=new_obstacle_position,
                    orientation=new_obstacle_orientation,
                    scale_factor=scale_factor,
                    enable_physics=True,
                    physics_material_name=obstacle["physics_material"]
                )
                candidate_prop_exists = True
            world.render()
            
            # check for overlap with existing props
            overlap = False
            for _, prop in existing_scene_props.items():
                overlap = check_object_overlap(candidate_prop.get_aabb(), prop.get_aabb(), margin=prop_buffer_margin)
                if overlap:
                    placement_attempt_counter += 1
                    break
        
            # if successful placement, add to existing_scene_props and break out of the loop
            if not overlap:
                placement_successful = True
                break
        
        if placement_successful:
            existing_scene_props.update(
                {"obstacle_" + obstacle["name"]: candidate_prop}
            )     