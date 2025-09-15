import copy

from pathlib import Path
from typing import List

import base64
import coacd
import io
import json
import numpy as np
import open3d as o3d
import torch

from openai import OpenAI
from PIL import Image, ImageDraw

from api_key import OPENAI_API_KEY
from src.physics.sim_entities import Mesh, Sphere
from src.splatting.datasets import get_dataset
from src.splatting.gaussian_model import GaussianModel
from src.utils.geometry_3d import densify_point_cloud, truncate_point_cloud, point_cloud_to_triangle_mesh
from src.utils.io import create_directory, get_unique_file_path, save_dict_to_json, save_dict_to_pkl
from src.utils.llm import query_llm
from src.utils.meshing import compute_multibody_mesh_volume, mesh_convex_decomposition, save_multibody_mesh_to_obj
from src.utils.rendering import get_render_settings, render_gaussian_model
from src.utils.splatting import refine_gaussian_model_segmentation
from src.vision_models.utils.detection import inflate_bboxes

SYSTEM_MESSAGE = "You are an assistant to an autonomous robot. Your job is to observe the robot's visual observations and answer its questions. The robot will provide its input query and relevant context under \"Input.\" It will provide more specific requirements pertaining to the query under \"Task.\" Please provide your response in the specified format."
QUERY_TEMPLATE = """Input:

The robot observes a scene and detects an object of interest. The image is provided with a bounding box indicating the object of interest. The robot has a preliminary annotation of the object's identity, and indicates it to be {annotation}, which may or may not be correct. The robot seeks verification of the object's identity and physical property estimation.

Task:

Verify the identity of the object inside the bounding box. If the annotation is accurate, confirm it. If it is inaccurate, provide the most appropriate object label. Furthermore, provide a general description of the object's identity, physical appearance, and purpose.

The robot requires an estimate of the object's physical properties to calculate and plan for environment interactions. First, determine the most appropriate material for the object. If the object appears to be composed of multiple materials or the material is indiscernible, please provide the most prevalent or representative material. Then, estimate the following physical properties of the material:

Density (kg/m^3)
Friction Coefficient
Coefficient of Restitution
Young’s Modulus (Pa)
Poisson’s Ratio

Please provide a single best numerical estimate for each physical property. The output should be structured as a JSON file, with the following fields:

-Annotation Accuracy
-Object Label
-General Description
-Material
-Density
-Friction Coefficient
-Coefficient of Restitution
-Young's Modulus
-Poisson's Ratio"""

def parse_output_string(input_string):
    """
    Converts the LLM output string into a dictionary.

    Args:
        input_string (str): The input string containing JSON data.
    
    Returns:
        dict: The parsed JSON data as a dictionary.
    """
    # remove the Markdown formatting
    json_string = input_string.strip('```json\n').strip('\n```')
    json_data = json.loads(json_string)

    return json_data


# def flatten_mesh_faces(mesh, n_percent=5, flatten_top=True, flatten_bottom=True):
#         vertices = np.asarray(mesh.vertices)
#         z = vertices[:, 2]
#         N = len(z)
#         sorted_indices = np.argsort(z)
#         # Flatten bottom
#         if flatten_bottom:
#             bottom_n = max(1, int(N * n_percent / 100))
#             bottom_indices = sorted_indices[:bottom_n]
#             flat_z = float(np.median(z[bottom_indices]))
#             vertices[bottom_indices, 2] = flat_z
#         # Flatten top
#         if flatten_top:
#             top_n = max(1, int(N * n_percent / 100))
#             top_indices = sorted_indices[-top_n:]
#             flat_z = float(np.median(z[top_indices]))
#             vertices[top_indices, 2] = flat_z
#         mesh.vertices = o3d.utility.Vector3dVector(vertices)
#         return mesh

from scipy.stats import mode

def flatten_mesh_faces(mesh, n_percent=10, flatten_top=True, flatten_bottom=True):
    vertices = np.asarray(mesh.vertices)
    z = vertices[:, 2]
    N = len(z)
    sorted_indices = np.argsort(z)
    
    # Flatten bottom
    if flatten_bottom:
        bottom_n = max(1, int(N * n_percent / 100))
        bottom_indices = sorted_indices[:bottom_n]
        bottom_zs = z[bottom_indices]
        # Use mode or median for robustness
        flat_z = float(mode(np.round(bottom_zs, 5))[0]) if len(bottom_zs) > 1 else bottom_zs[0]
        vertices[bottom_indices, 2] = flat_z
    
    # Flatten top
    if flatten_top:
        top_n = max(1, int(N * n_percent / 100))
        top_indices = sorted_indices[-top_n:]
        top_zs = z[top_indices]
        flat_z = float(mode(np.round(top_zs, 5))[0]) if len(top_zs) > 1 else top_zs[0]
        vertices[top_indices, 2] = flat_z
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


class Materializer:
    def __init__(
        self, 
        config,
        gaussian_model,
    ):
        self.config = config
        

        # prepare output directory
        run_name = self.config["dataset_config"]["dataset_path"].split("/")[-1] 
        self.run_directory = Path(self.config["save_directory"]) / run_name
        create_directory(self.run_directory / "entities", overwrite=True)

        # load dataset
        dataset_config = self.config["dataset_config"]
        dataset_config = copy.deepcopy(dataset_config)
        dataset_config["downsample"] = True # TODO: clean this
        self.dataset = get_dataset(dataset_config["dataset"])(dataset_config)

        # gaussian model
        self.gaussian_model = gaussian_model
        self.entity_labels = self.dataset.get_annotations(0)
        self.gaussian_point_classes = refine_gaussian_model_segmentation(
            self.gaussian_model,
            self.config,
            self.entity_labels,
        )

        # save full Gaussian model
        self.gaussian_model.save_ply(self.run_directory / "gaussian_model.ply")

        # save global gaussian model with colored segments; used for debugging visualization 
        from src.utils.math_utils import rgb_to_sh
        # generate random set of N distinct colors:
        N = len(self.entity_labels)
        colors = np.random.rand(N, 3)
        colors = np.clip(colors, 0.01, 0.99)
        # color mask for segmented gaussians
        segmented_f_dc = (torch.zeros((len(self.gaussian_point_classes), 3, (gaussian_model.max_sh_degree + 1) ** 2)).float().cuda())
        for i in range(N):
            mask = (self.gaussian_point_classes == i)
            segmented_f_dc[mask, :3, 0] = rgb_to_sh(torch.tensor(colors[i]).float().cuda())
            segmented_f_dc[:, 3:, 1:] = 0.0
        gaussian_model._features_dc = segmented_f_dc
        gaussian_model.save_ply(self.run_directory / "segmented_gaussian_model.ply")
        
    def run(self):
        # # begin by refining Gaussian model segmentation
        # entity_labels = self.dataset.get_annotations(0)
        # point_classes = refine_gaussian_model_segmentation(
        #     self.gaussian_model,
        #     self.config,
        #     entity_labels,
        # )
        num_objects = len(self.dataset.get_bboxes(0))

        for i, label in enumerate(self.entity_labels):
            print("---------")
            print("Materializing entity", i, "with label", label)
            entity_directory = self.run_directory / "entities" / f"{label}_class_id_{i}"
            create_directory(entity_directory, overwrite=True)

            # create metadata file
            entity_info = {
                "label": label,
                "class_id": i,
                "directory": str(entity_directory),
            }
            save_dict_to_json(entity_info, "metadata.json", directory=entity_directory)

            # extract Gaussian model segment corresponding to entity, and save
            self.segment_entity_gaussian_model(entity_info)

            if i >= num_objects:
                continue
            
            # convert segmented Gaussian model to mesh, and save
            self.construct_entity_mesh(entity_info)

            # obtain entity's material properties, and save
            self.get_entity_material_properties(entity_info)



    def segment_entity_gaussian_model(self, entity_info):
        entity_label = entity_info["label"]
        entity_id = entity_info["class_id"]
        entity_directory = Path(entity_info["directory"])

        # save Gaussians corresponding to particular entity of interest
        cls_indices = torch.where(self.gaussian_point_classes == entity_id)[0].cpu()
        mask = torch.zeros(self.gaussian_model.get_xyz().shape[0], dtype=torch.bool)
        mask[cls_indices] = True
        segmented_model_path = entity_directory / "gaussian_model.ply"
        self.gaussian_model.save_ply(segmented_model_path, mask)
        print(f"Saved Gaussian model segment for {entity_label}")

    def get_entity_material_properties(self, entity_info):
        entity_label = entity_info["label"]
        entity_id = entity_info["class_id"]
        entity_directory = Path(entity_info["directory"])

        # create openai client
        client = OpenAI(api_key=OPENAI_API_KEY)

        # get data for reference image
        img = self.dataset.get_color_image(0)
        bboxes = self.dataset.get_bboxes(0)
        inflated_bboxes = inflate_bboxes(
            bboxes,
            n_pix_horizontal=5,
            n_pix_vertical=5,
        )

        if entity_id >= len(bboxes):
            return

        # draw detection's bounding box on image
        bbox = inflated_bboxes[entity_id].astype(int)
        xmin, ymin, xmax, ymax = bbox
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle(bbox.tolist(), outline="red", width=3)

        # send image to in-memory byte stream and encode to base64
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format="jpeg")
        img_byte_arr.seek(0)
        img_bytes = img_byte_arr.getvalue()
        base64_img = base64.b64encode(img_bytes).decode("utf-8")

        # query
        system_message = SYSTEM_MESSAGE
        query_message = QUERY_TEMPLATE.format(annotation=entity_label)
        response = query_llm(
            client,
            system_message=system_message,
            user_message=query_message,
            user_images=[base64_img],
        )
        print("Entity Material Properties:")
        print(response.output_text)

        # save results
        # full OpenAI response (debugging)
        save_dict_to_pkl(
            response,
            "llm_response_full" + ".pkl",
            directory=entity_directory,
        )

        # parsed object properties and materials
        object_properties_dict = parse_output_string(response.output_text)
        save_dict_to_json(
            object_properties_dict,
            "materials" + ".json",
            directory=entity_directory,
        )

        # save crop of detection (visualization/debugging)
        pil_img = Image.fromarray(img)
        crop = pil_img.crop((xmin, ymin, xmax, ymax))
        crop.save((entity_directory / "image").with_suffix(".jpeg"))

    def construct_entity_mesh(self, entity_info):
        entity_label = entity_info["label"]
        entity_id = entity_info["class_id"]
        entity_directory = Path(entity_info["directory"])

        meshing_config = self.config["meshing"]

        # load Gaussian model corresponding to entity of interest
        gaussian_model = GaussianModel(0)
        gaussian_model.load_ply(entity_directory / "gaussian_model.ply")
        
        # extract point cloud
        if meshing_config["densify_point_cloud"]:
            densified_point_cloud = densify_point_cloud(
                gaussian_centers=gaussian_model.get_xyz(), 
                gaussian_rotations=gaussian_model.get_rotation(),
                gaussian_scales=gaussian_model.get_scaling(),
                density=25000.0,#50000.0,
                control_points_extent=2.5
            ).detach().cpu().numpy()
            point_cloud = gaussian_model.get_xyz().detach().cpu().numpy()
            point_cloud = np.vstack((point_cloud, densified_point_cloud))
        else:
            point_cloud = gaussian_model.get_xyz().detach().cpu().numpy()
            normals = gaussian_model.get_normals().detach().cpu().numpy()

        # point cloud clean up
        truncated_points = truncate_point_cloud(
            point_cloud, 
            x_bounds=meshing_config["x_bounds"],
            y_bounds=meshing_config["y_bounds"],
            z_bounds=meshing_config["z_bounds"], 
        )

        # visualize point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(truncated_points)
        o3d.visualization.draw_geometries([pcd], mesh_show_back_face=True, mesh_show_wireframe=True)

        if "ball" in entity_label:
            sphere = Sphere.fit(truncated_points, threshold=0.001, max_iterations=1000)
            # sphere.name = entity_label
            sphere.save(entity_directory / "mesh")

        else:
            # if "box" in entity_label:
            #     mesh_vertices, surface_triangles = point_cloud_to_triangle_mesh(truncated_points, max_edge_length=0.05)
            # else:
            mesh_vertices, surface_triangles = point_cloud_to_triangle_mesh(truncated_points, max_edge_length=meshing_config["max_tetra_edge_length"], adaptive_edge_pruning=meshing_config["adaptive_edge_pruning"])

            # smooth mesh
            if meshing_config["smooth_mesh"]:
                # create o3d mesh object
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
                mesh.triangles = o3d.utility.Vector3iVector(surface_triangles)

                # if "box" in entity_label:
                # identify connected components, keep only largest to avoid spurious mesh components
                triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)
                cluster_area = np.asarray(cluster_area)

                cluster_idces_to_remove = np.where(cluster_n_triangles < cluster_n_triangles.max())[0]
                triangles_remove_mask = np.isin(triangle_clusters, cluster_idces_to_remove).tolist()
                mesh.remove_triangles_by_mask(triangles_remove_mask)

                # smooth
                mesh = mesh.filter_smooth_taubin(number_of_iterations=meshing_config["taubin_iterations"])

                # decimate
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=750)

                # compute normals for plotting
                mesh.orient_triangles()
                mesh.compute_vertex_normals()
                # mesh.orie
                o3d.visualization.draw_geometries([mesh], mesh_show_back_face=False, mesh_show_wireframe=True)

            o3d.io.write_triangle_mesh(str(entity_directory / "mesh.obj"), mesh)

            # import pybullet as p
            # p.vhacd(
            #     str(entity_directory / "mesh.obj"), 
            #     str(entity_directory / "mesh_decomposed_vhacd.obj"), 
            #     str(entity_directory / "vhacd_log.txt"),
            #     maxNumVerticesPerCH=256,
            #     mode=1,
            # ),

            convex_decomposition = mesh_convex_decomposition(mesh)
            save_multibody_mesh_to_obj(str(entity_directory / "mesh_decomposed.obj"), convex_decomposition)

            # load and visualize the mesh
            mesh = o3d.io.read_triangle_mesh(str(entity_directory / "mesh_decomposed.obj"))
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            o3d.visualization.draw_geometries([mesh], mesh_show_back_face=False, mesh_show_wireframe=True, point_show_normal=True)

        print(f"Saved mesh for {entity_label}")

    # def mesh_alignment(self):
    #     num_objects = len(self.dataset.get_bboxes(0))

    #     # load meshes
    #     meshes = []
    #     for i, label in enumerate(self.entity_labels):
    #         entity_directory = self.run_directory / "entities" / f"{label}_class_id_{i}"
            
    #         mesh_path = entity_directory / "mesh.obj"

    #         if not mesh_path.exists():
    #             print(f"Mesh not found for {label}")
    #             continue
    #         mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    #         meshes.append(mesh)
    #         breakpoint()
    #     breakpoint()

    #     # order meshes by height (use mid-z coordinate of aabb)
    #     mesh_heights = []
    #     for mesh in meshes:
    #         aabb = mesh.get_axis_aligned_bounding_box()
    #         mid_z = (aabb.max_bound[2] + aabb.min_bound[2]) / 2
    #         mesh_heights.append(mid_z)
    #     sorted_indices = np.argsort(mesh_heights)
    #     meshes = [meshes[i] for i in sorted_indices]
    #     print("Objects in order of height:", [self.entity_labels[i] for i in sorted_indices])

    #     # align meshes (flatten top and bottom planes)
    #     mesh_heights = []
    #     for mesh in meshes:
    #         # get lowest z
    #         aabb = mesh.get_axis_aligned_bounding_box()
    #         min_z = aabb.min_bound[2]

    #         # determine if it's touching the ground
    #         touching_ground = True if min_z < 0.05 else False

    #         # only flatten bottom faces if not touching ground
    #         if not touching_ground:

    

    def mesh_alignment(self, n_percent=5, gap=0.01):
        num_objects = len(self.dataset.get_bboxes(0))

        # load meshes
        meshes = []
        mesh_labels = []
        for i, label in enumerate(self.entity_labels):
            entity_directory = self.run_directory / "entities" / f"{label}_class_id_{i}"
            mesh_path = entity_directory / "mesh.obj"
            if not mesh_path.exists():
                print(f"Mesh not found for {label}")
                continue
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            meshes.append(mesh)
            mesh_labels.append(label)

        # order meshes by height (use mid-z coordinate of aabb)
        mesh_heights = []
        for mesh in meshes:
            aabb = mesh.get_axis_aligned_bounding_box()
            mid_z = (aabb.max_bound[2] + aabb.min_bound[2]) / 2
            mesh_heights.append(mid_z)
        sorted_indices = np.argsort(mesh_heights)
        meshes = [meshes[i] for i in sorted_indices]
        mesh_labels = [mesh_labels[i] for i in sorted_indices]
        print("Objects in order of height:", mesh_labels)

        # Flatten and stack meshes
        prev_top_z = None
        for idx, mesh in enumerate(meshes):
            aabb = mesh.get_axis_aligned_bounding_box()
            min_z = aabb.min_bound[2]
            max_z = aabb.max_bound[2]
            # Determine if touching ground
            touching_ground = min_z < 0.05
            # Flatten faces
            mesh = flatten_mesh_faces(
                mesh,
                n_percent=n_percent,
                flatten_top=True,
                flatten_bottom=not touching_ground
            )
            # Align bottom to previous top
            if prev_top_z is not None and not touching_ground:
                vertices = np.asarray(mesh.vertices)
                new_min_z = np.min(vertices[:, 2])
                dz = prev_top_z + gap - new_min_z
                mesh.translate([0, 0, dz])
            # Update prev_top_z for next mesh
            prev_top_z = np.max(np.asarray(mesh.vertices)[:, 2])

            # visualize mesh
            mesh.orient_triangles()
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh], mesh_show_back_face=False, mesh_show_wireframe=True)
            

            # Optionally, save the aligned mesh
            entity_directory = self.run_directory / "entities" / f"{mesh_labels[idx]}_class_id_{sorted_indices[idx]}"
            print("Saving aligned mesh for", mesh_labels[idx])
            print("Save path: ", str(entity_directory / f"mesh_aligned.obj"))
            o3d.io.write_triangle_mesh(str(entity_directory / f"mesh_aligned.obj"), mesh)





            
            


