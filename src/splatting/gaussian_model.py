from pathlib import Path
from typing import Optional, Union

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import pytorch3d.ops

from plyfile import PlyData, PlyElement
from pytorch3d.transforms import quaternion_to_matrix
from torch import nn

from simple_knn._C import distCUDA2 # needs to be imported after torch
from src.splatting.layers import ClassificationLayer
from src.utils.math_utils import rgb_to_sh
from src.utils.training import get_exponential_lr_scheduler

EXTRA_FEATURE_DIM = 16 # TODO put this in a better place


class GaussianModel:
    def __init__(
        self, 
        sh_degree: int = 3, 
        isotropic: bool = False,
        include_point_features: bool = False,
    ) -> None:
        self.gaussian_param_names = [
            "active_sh_degree",
            "xyz",
            "features_dc",
            "features_rest",
            "features_extra",
            "scaling",
            "rotation",
            "opacity",
            "opacity_point_features",
            "max_radii2D",
            "xyz_gradient_accum",
            "denom",
            "spatial_lr_scale",
            "optimizer",
        ]

        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree  # temp
        self.isotropic = isotropic
        self.include_point_features = include_point_features

        self._xyz = torch.empty(0).cuda()
        self._features_dc = torch.empty(0).cuda()
        self._features_rest = torch.empty(0).cuda()
        self._features_extra = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0, 4).cuda()
        self._opacity = torch.empty(0).cuda()
        self._opacity_point_features = torch.empty(0).cuda()
        self._polarity = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.splat_optimizer = None
        self.feature_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 1

        # setup functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit
        self.rotation_activation = torch.nn.functional.normalize

        self.point_classifier = None

    def point_classifier_setup(self, num_classes: int):
        self.point_classifier = ClassificationLayer(EXTRA_FEATURE_DIM, num_classes).cuda()
        self.feature_optimizer.add_param_group(
            {"params": self.point_classifier.parameters(), "lr": 0.01, "name": "point_classifier"}
        )

    def add_points(self, point_cloud: o3d.geometry.PointCloud, global_scale_init=True):
        """
        Adds a point cloud to the Gaussian model. This involves creating the Gaussian parameters for the new points, registering
        them with the optimizer, and adding them to the model.

        Args:
            point_cloud (o3d.geometry.PointCloud): Point cloud to add.
            global_scale_init (bool): If True, initializes the scale of the new points to the global scale.
        """

        # (preprocessing step) generate new parameters from the new point cloud
        new_parameter_dict = self._point_cloud_to_params(point_cloud, global_scale_init)
        
        # similarly, add segmentation features if needed
        if self.include_point_features:
            new_feature_parameter_dict = {}
            new_feature_parameter_dict["f_extra"] = nn.Parameter(
                torch.nn.functional.normalize(
                    torch.ones(
                        (
                            np.asarray(point_cloud.points).shape[0], 
                            EXTRA_FEATURE_DIM
                        ), device="cuda").requires_grad_(True),
                    p=2,
                    dim=1
                )
            )
            new_feature_parameter_dict["opacity_point_features"] = nn.Parameter(
                self.inverse_opacity_activation(
                    1. * torch.ones((np.asarray(point_cloud.points).shape[0], 1), dtype=torch.float, device="cuda")
                ).requires_grad_(True)
            )            

        # register new parameters with optimizer and merge all model parameters (newly added and existing)
        merged_splat_parameters = self._add_params_to_optimizer(new_parameter_dict, self.splat_optimizer)
        if self.include_point_features:
            merged_feature_parameters = self._add_params_to_optimizer(new_feature_parameter_dict, self.feature_optimizer)
            merged_splat_parameters.update(merged_feature_parameters)

        # update model
        self._update_model(merged_splat_parameters)

    def remove_points(self, remove_mask):
        # remove corresponding parameters from optimizer
        pruned_parameters = self._remove_parameters_from_optimizer(remove_mask, self.splat_optimizer)
        if self.include_point_features:
            pruned_parameters.update(self._remove_parameters_from_optimizer(remove_mask, self.feature_optimizer))

        # update model with pruned points
        self._xyz = pruned_parameters["xyz"]
        self._features_dc = pruned_parameters["f_dc"]
        self._features_rest = pruned_parameters["f_rest"]
        self._scaling = pruned_parameters["scaling"]
        self._rotation = pruned_parameters["rotation"]
        self._opacity = pruned_parameters["opacity"]
        self._polarity = pruned_parameters["polarity"]

        if self.include_point_features:
            self._features_extra = pruned_parameters["f_extra"]
            self._opacity_point_features = pruned_parameters["opacity_point_features"]

        keep_mask = ~remove_mask
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        self.denom = self.denom[keep_mask]
        self.max_radii2D = self.max_radii2D[keep_mask]

    def get_size(self):
        return self._xyz.shape[0]

    def get_xyz(self):
        return self._xyz
    
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    def get_point_features(self):
        return self._features_extra    

    def get_scaling(self):
        if self.isotropic:
            scale = self.scaling_activation(self._scaling)[:, 0:1]
            scales = scale.repeat(1, 3)
            return scales
        else:
            return self.scaling_activation(self._scaling)
    
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_opacity_point_features(self):
        return self.opacity_activation(self._opacity_point_features) 
    
    def get_normals(self):
        try:
            R = quaternion_to_matrix(self.get_rotation())
            indices = torch.argmin(self.get_scaling(), dim=1)
            oriented_normals = R[torch.arange(R.shape[0]), :, indices] * self._polarity
            normals = F.normalize(oriented_normals, p=2, dim=1)
        except:
            breakpoint()
        return normals
    
    def capture_dict(self):
        param_dict = {
            "active_sh_degree": self.active_sh_degree,
            "isotropic": self.isotropic,
            "include_point_features": self.include_point_features,
            "xyz": self._xyz.clone().detach().cpu(),
            "features_dc": self._features_dc.clone().detach().cpu(),
            "features_rest": self._features_rest.clone().detach().cpu(),
            "features_extra": self._features_extra.clone().detach().cpu(),
            "scaling": self._scaling.clone().detach().cpu(),
            "rotation": self._rotation.clone().detach().cpu(),
            "opacity": self._opacity.clone().detach().cpu(),
            "opacity_point_features": self._opacity_point_features.clone().detach().cpu(),
            "polarity": self._polarity.clone().detach().cpu(),
            "max_radii2D": self.max_radii2D.clone().detach().cpu(),
            "xyz_gradient_accum": self.xyz_gradient_accum.clone().detach().cpu(),
            "denom": self.denom.clone().detach().cpu(),
            "spatial_lr_scale": self.spatial_lr_scale,
            "splat_optimizer": self.splat_optimizer.state_dict(),
            "feature_optimizer": self.feature_optimizer.state_dict(),
        }

        if self.point_classifier is not None:
            param_dict["point_classifier"] = self.point_classifier.state_dict()

        return param_dict

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        l.append("polarity")
        return l
            
    def set_optimizer_zero_grad(self):
        self.splat_optimizer.zero_grad(set_to_none=True)
        if self.include_point_features:
            self.feature_optimizer.zero_grad(set_to_none=True)
    
    def training_setup(self, optimization_params):
        self._splat_training_setup(optimization_params)
        if self.include_point_features:
            self._feature_training_setup(optimization_params)

    def load_parameters(self, parameter_dict: dict):
        self.max_sh_degree = parameter_dict["active_sh_degree"]
        self.active_sh_degree = parameter_dict["active_sh_degree"]
        self.isotropic = parameter_dict["isotropic"]
        self.include_point_features = parameter_dict["include_point_features"]

        self._xyz = parameter_dict["xyz"].cuda()
        self._features_dc = parameter_dict["features_dc"].cuda()
        self._features_rest = parameter_dict["features_rest"].cuda()
        self._features_extra = parameter_dict["features_extra"].cuda()
        self._scaling = parameter_dict["scaling"].cuda()
        self._rotation = parameter_dict["rotation"].cuda()
        self._opacity = parameter_dict["opacity"].cuda()
        self._opacity_point_features = parameter_dict["opacity_point_features"].cuda()
        self._polarity = parameter_dict["polarity"].cuda()
        self.max_radii2D = parameter_dict["max_radii2D"].cuda()
        self.xyz_gradient_accum = parameter_dict["xyz_gradient_accum"].cuda()
        self.denom = parameter_dict["denom"].cuda()
        self.splat_optimizer = None
        self.feature_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = parameter_dict["spatial_lr_scale"]

        # setup functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit
        self.rotation_activation = torch.nn.functional.normalize

        # load point classifier if it exists
        if "point_classifier" in parameter_dict:
            self.point_classifier = ClassificationLayer(EXTRA_FEATURE_DIM, parameter_dict["point_classifier"]["linear.weight"].shape[0]).cuda()
            self.point_classifier.load_state_dict(parameter_dict["point_classifier"])


    def save_ply(self, save_path: Union[str, Path], mask: Optional[torch.Tensor] = None):
        if isinstance(save_path, str):
            save_path = Path(save_path)

        xyz = self._xyz.detach().cpu().numpy()
        normals = self.get_normals().detach().cpu().numpy()
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        if self.isotropic:
            # tile into shape (P, 3)
            scale = np.tile(self._scaling.detach().cpu().numpy()[:, 0].reshape(-1, 1), (1, 3))
        else:
            scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        polarity = self._polarity.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        if mask is not None:
            elements = np.empty(xyz[mask].shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz[mask], normals[mask], f_dc[mask], f_rest[mask], opacities[mask], scale[mask], rotation[mask], polarity[mask]), axis=1)
        else:
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, polarity), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(save_path)

    def load_ply(self, ply_path):
        plydata = PlyData.read(ply_path)

        xyz = np.stack((
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),
                axis=1)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        polarity = np.asarray(plydata.elements[0]["polarity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._polarity = nn.Parameter(torch.tensor(polarity, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        
    def _add_params_to_optimizer(self, new_parameter_dict, optimizer):
        # initialize merged (new and existing) parameter dict
        merged_params = {}

        # iterate through optimizer parameter groups and update
        for group in optimizer.param_groups:
            if group["name"] == "point_classifier":
                continue
            assert len(group["params"]) == 1
            param_name = group["name"]
            new_params = new_parameter_dict[param_name]

            stored_params = optimizer.state.get(group["params"][0], None)
            if stored_params is not None:
                # if the parameter group already exists, initialize optimizer state for new parameters
                stored_params["exp_avg"] = torch.cat(
                    (stored_params["exp_avg"], torch.zeros_like(new_params)), dim=0
                )
                stored_params["exp_avg_sq"] = torch.cat(
                    (stored_params["exp_avg_sq"], torch.zeros_like(new_params)), dim=0
                )

                # delete and replace existing optimizer state
                del optimizer.state[group["params"][0]] #TODO: check if this is needed still - possibly prevents memory leaks.
                group["params"][0] = nn.Parameter( # add new parameters to parameter group
                    torch.cat((group["params"][0], new_params), dim=0).requires_grad_(True)
                )                
                optimizer.state[group["params"][0]] = stored_params
                
            else:
                # add new parameters
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], new_params), dim=0).requires_grad_(True)
                )
            
            # add new parameter set to dictionary
            merged_params[param_name] = group["params"][0]
        
        return merged_params
    
    def _point_cloud_to_params(self, point_cloud: o3d.geometry.PointCloud, global_scale_init=True) -> dict:
        """
        Create new Gaussian parameters corresponding to new points from the input point cloud.
        
        Args:
            point_cloud (o3d.geometry.PointCloud): Point cloud to add.
            global_scale_init (bool): If True, initializes the scale of the new points to the global scale.
        Returns:
            dict: Dictionary containing the new Gaussian parameters.
        """
        # convert point cloud to torch
        fused_point_cloud = torch.tensor(np.asarray(point_cloud.points)).float().cuda()

        # set features
        fused_color = rgb_to_sh(torch.tensor(np.asarray(point_cloud.colors)).float().cuda())
        features = (torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda())
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # scale initialization for Gaussian points based on mean distance to neighboring Gaussian points
        # if global_scale_init is True, neighbors are all particles; otherwise, only the new point cloud points are considered
        if global_scale_init:
            global_points = torch.cat(
                (
                    self.get_xyz(), 
                    torch.from_numpy(np.asarray(point_cloud.points)).float().cuda()
                )
            )
            dist2 = torch.clamp_min(distCUDA2(global_points), 0.0000001)
            dist2 = dist2[self.get_size():]
        else:
            point_cloud_points = torch.from_numpy(np.asarray(point_cloud.points)).float().cuda()
            dist2 = torch.clamp_min(distCUDA2(point_cloud_points), 0.0000001)
        scale_factor = 1.0
        scales = torch.log(scale_factor * torch.sqrt(dist2))[..., None].repeat(1, 3)

        # rotation initialization (quaternion representation with w as the first element)
        rotations = torch.zeros((fused_point_cloud.shape[0], 4), dtype=torch.float, device="cuda")
        rotations[:, 0] = 1.0

        # opacity initialization (uniformly initialized to 0.5 for all points)
        opacities = self.inverse_opacity_activation(
            0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        # normal polarity initialization
        polarities = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        # add new points as model parameters
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rotations.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))
        new_polarity = nn.Parameter(polarities.requires_grad_(True))

        new_parameter_dict = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "opacity": new_opacity,
            "polarity": new_polarity,
        }

        return new_parameter_dict

    def _remove_parameters_from_optimizer(self, remove_mask, optimizer):
        # initialize pruned parameter dict (output dict)
        pruned_params = {}
        keep_mask = ~remove_mask

        # iterate through optimizer parameter groups and update
        for group in optimizer.param_groups:
            if group["name"] == "point_classifier":
                continue
            assert len(group["params"]) == 1 # TODO: maybe this should be removed

            # remove optimizer state for removed parameters
            stored_params = optimizer.state.get(group["params"][0], None)
            if stored_params is not None:
                stored_params["exp_avg"] = stored_params["exp_avg"][keep_mask]
                stored_params["exp_avg_sq"] = stored_params["exp_avg_sq"][keep_mask]

                # delete and replace existing optimizer state
                del optimizer.state[group["params"][0]] #TODO: check if this is needed still - possibly prevents memory leaks.
                group["params"][0] = nn.Parameter((group["params"][0][keep_mask].requires_grad_(True)))
                optimizer.state[group["params"][0]] = stored_params
            else:
                # remove parameters
                group["params"][0] = nn.Parameter(group["params"][0][keep_mask].requires_grad_(True))
            # add pruned parameter set to dictionary
            pruned_params[group["name"]] = group["params"][0]
        
        return pruned_params

    def _update_model(self, merged_parameters: dict) -> None:
        """
        Update the Gaussian model with new parameters.

        Args:
            merged_parameters (dict): Dictionary containing the new Gaussian parameters.
        """        
        # update Gaussian model attributes with merged parameters
        self._xyz = merged_parameters["xyz"]
        self._features_dc = merged_parameters["f_dc"]
        self._features_rest = merged_parameters["f_rest"]
        self._scaling = merged_parameters["scaling"]
        self._rotation = merged_parameters["rotation"]
        self._opacity = merged_parameters["opacity"]
        self._polarity = merged_parameters["polarity"]

        if self.include_point_features:
            self._features_extra = merged_parameters["f_extra"]
            self._opacity_point_features = merged_parameters["opacity_point_features"]

        # TODO: check these shapes
        self.xyz_gradient_accum = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz().shape[0]), device="cuda")

    def _feature_training_setup(self, optimization_params):
        # define optimization parameter groups and associated optimizer arguments
        params = [
            {"params": [self._features_extra], "lr": optimization_params.feature_extra_lr, "name": "f_extra"},
            {"params": [self._opacity_point_features], "lr": optimization_params.opacity_lr, "name": "opacity_point_features"},
        ]

        # create optimizer
        self.feature_optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

    def _splat_training_setup(self, optimization_params):
        self.percent_dense = optimization_params.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz().shape[0], 1), device="cuda") # what is this?
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device="cuda") # what is this?

        # define optimization parameter groups and associated optimizer arguments
        params = [
            {"params": [self._xyz], "lr": optimization_params.position_lr_init, "name": "xyz"},
            {"params": [self._features_dc], "lr": optimization_params.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": optimization_params.feature_lr/20., "name": "f_rest"},
            {"params": [self._opacity], "lr": optimization_params.opacity_lr, "name": "opacity"},
            {"params": [self._rotation], "lr": optimization_params.rotation_lr, "name": "rotation"},
            {"params": [self._scaling], "lr": optimization_params.scaling_lr, "name": "scaling"},
            {"params": [self._polarity], "lr": optimization_params.polarity_lr, "name": "polarity"},
        ]

        # create optimizer
        self.splat_optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

        # learning rate scheduler
        self.xyz_scheduler = get_exponential_lr_scheduler(
            lr_init=optimization_params.position_lr_init * self.spatial_lr_scale,
            lr_final=optimization_params.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=optimization_params.position_lr_delay_mult, # TODO: doesn't do much since lr_delay_steps is usually 0 or close to it
            max_steps=optimization_params.position_lr_max_steps,
        )