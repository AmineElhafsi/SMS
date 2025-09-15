from typing import List

import coacd
import numpy as np
import open3d as o3d

def compute_multibody_mesh_volume(file_path: str) -> float:
    """
    Compute the volume of a multi-body mesh from an OBJ file.
    
    """
    vertices = []
    faces = []
    components = []
    vertex_offset = 0 # tracks global vertex offset

    # Parse the OBJ file to extract vertices and faces for each component
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # vertices
                vertices.append(list(map(float, line.split()[1:])))
            elif line.startswith('f '):  # faces
                face = [int(idx.split('/')[0]) - 1 for idx in line.split()[1:]]
                faces.append(face)
            elif line.startswith('o ') or line.startswith('g '):  # new component
                if vertices and faces:
                    local_faces = np.array(faces) - vertex_offset
                    components.append((np.array(vertices), local_faces))
                    # components.append((np.array(vertices), np.array(faces)))

                # Reset vertices and faces for the next component
                vertex_offset += len(vertices)
                vertices = []
                faces = []

        # ddd the last component
        if vertices and faces:
            local_faces = np.array(faces) - vertex_offset
            components.append((np.array(vertices), local_faces))

    # compute volume
    volumes = []
    for vertices, faces in components:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        vol = mesh.get_volume()
        volumes.append(vol)
    total_volume = sum(volumes)
    
    return total_volume

def mesh_convex_decomposition(mesh: o3d.geometry.TriangleMesh) -> List[o3d.geometry.TriangleMesh]:
    """
    Decomposes a mesh into convex parts using the COACD algorithm.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh to be decomposed.
    Returns:
        o3d.geometry.TriangleMesh: The decomposed mesh parts.
    """
    # decompose mesh
    coacd_mesh = coacd.Mesh(
        np.asarray(mesh.vertices), 
        np.asarray(mesh.triangles)
    )

    result = coacd.run_coacd(
        coacd_mesh,
        threshold=0.075,
        max_convex_hull=-1,
        preprocess_mode="auto",
        preprocess_resolution=30,
        resolution=2000,
        mcts_nodes=20,
        mcts_iterations=1000,
        mcts_max_depth=1,
        pca=False,
        merge=False,
        decimate=True,
        max_ch_vertex=256,
        extrude=False,
        extrude_margin=0.01,
        apx_mode="ch",
        seed=0,
    )

    # merge
    convex_decomposition = []
    volume = 0
    for vs, fs in result:
        mesh_part = o3d.geometry.TriangleMesh()
        mesh_part.vertices = o3d.utility.Vector3dVector(vs)
        mesh_part.triangles = o3d.utility.Vector3iVector(fs)
        mesh_part.compute_vertex_normals()
        mesh_part.remove_degenerate_triangles()
        mesh_part.remove_duplicated_triangles()
        mesh_part.remove_duplicated_vertices()
        mesh_part.remove_non_manifold_edges()
        volume += mesh_part.get_volume()
        convex_decomposition.append(mesh_part)

    return convex_decomposition

def save_multibody_mesh_to_obj(output_path, mesh_parts):
    """
    Save multiple Open3D meshes into a single .obj file.

    Args:
        mesh_parts (list): List of Open3D TriangleMesh objects.
        output_path (str): Path to save the .obj file.
    """
    with open(output_path, 'w') as obj_file:
        vertex_offset = 0

        for i, mesh in enumerate(mesh_parts):
            # Validate the mesh
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            if len(vertices) == 0:
                print(f"Warning: Mesh part {i + 1} has no vertices. Skipping.")
                continue

            if len(triangles) == 0:
                print(f"Warning: Mesh part {i + 1} has no triangles. Skipping.")
                continue

            # Check for invalid triangle indices
            for triangle in triangles:
                if any(idx >= len(vertices) or idx < 0 for idx in triangle):
                    raise ValueError(
                        f"Invalid triangle indices in mesh part {i + 1}. "
                        f"Triangle: {triangle}, Number of vertices: {len(vertices)}"
                    )

            # Write group name for each mesh part
            obj_file.write(f"# Mesh part {i + 1}\n")
            obj_file.write(f"o part_{i + 1}\n")

            # Write vertices
            for vertex in vertices:
                obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            # Write faces (adjust indices with vertex offset)
            for triangle in triangles:
                # OBJ format uses 1-based indexing
                obj_file.write(
                    f"f {triangle[0] + 1 + vertex_offset} "
                    f"{triangle[1] + 1 + vertex_offset} "
                    f"{triangle[2] + 1 + vertex_offset}\n"
                )

            # Update vertex offset
            vertex_offset += len(vertices)

    print(f"Saved multibody mesh to {output_path}")