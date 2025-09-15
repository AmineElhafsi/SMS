import numpy as np
import open3d as o3d
import pyransac3d as pyrsc


def load_sim_object(file_path):
    data = np.load(file_path)
    entity_type = data['object_type'].item()  # Convert numpy string to Python string

    if entity_type == "sphere":
        return Sphere.load(file_path)
    elif entity_type == "mesh":
        return Mesh.load(file_path)
    else:
        raise ValueError(f"Unknown object type: {entity_type}")
    

class SimEntity:
    def __init__(self, entity_type, name):
        self.entity_type = entity_type
        self.name = name

    def save(self, file_path):
        raise NotImplementedError("Save method not implemented.")
    
    @classmethod
    def load(cls, file_path):
        raise NotImplementedError("Load method not implemented.")
    
    def initialize_in_simulation(self, simulation):
        raise NotImplementedError("Initialize method not implemented.")


class Sphere(SimEntity):
    def __init__(self, center, radius, name=None):
        super().__init__(entity_type="sphere", name=name)
        self.center = center
        self.radius = radius
    
    def save(self, file_path):
        np.savez(file_path, object_type=self.entity_type, name=self.name, center=self.center, radius=self.radius)

    @classmethod
    def load(cls, file_path):
        data = np.load(file_path, allow_pickle=True)
        return cls(data["center"], data["radius"], data["name"].item())

    @staticmethod
    def fit(points, threshold, max_iterations):
        sphere_ransac = pyrsc.Sphere()
        center, radius, inliers = sphere_ransac.fit(pts=points, thresh=threshold, maxIteration=max_iterations)
        print("Center: ", center)
        print("Radius: ", radius)
        return Sphere(center, radius)
    

class Mesh(SimEntity):
    def __init__(self, vertices, triangles, name=None):
        super().__init__(entity_type="mesh", name=name)
        self.vertices = vertices
        self.triangles = triangles

    def save(self, file_path):
        np.savez(file_path, object_type=self.entity_type, name=self.name, vertices=self.vertices, triangles=self.triangles)

    @classmethod
    def load(cls, file_path):
        data = np.load(file_path)
        return cls(data["vertices"], data["triangles"], data["name"].item())