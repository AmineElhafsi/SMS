import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from src.splatting.gaussian_model import GaussianModel
from src.splatting.data_logging import Logger
from src.splatting.parameters import OptimizationParams
from src.utils.camera import depth_image_to_pointcloud, world_points_to_pixel_coordinates
from src.utils.evaluation import psnr
from src.utils.metrics import contrastive_clustering_loss, isotropic_loss, l1_loss, opacity_entropy_regularization_loss, spatial_regularization_loss, ssim
from src.utils.rendering import get_render_settings, render_gaussian_model, render_gaussian_model_features, render_gaussian_model_normals
from src.utils.segmentation import compute_mean_segmentation_mask_features
from src.utils.splatting import (compute_camera_frustum_corners, compute_frustum_point_ids, 
                                 create_point_cloud, geometric_edge_mask, keyframe_optimization_sampling_distribution, 
                                 orient_gaussian_normals_toward_camera, sample_control_points, sample_pixels_based_on_gradient, 
                                 select_new_point_ids)
from src.utils.utils import numpy_to_torch, numpy_to_point_cloud, torch_to_numpy

import matplotlib.pyplot as plt

def visualize_matrix(matrix, name, x_labels=None, y_labels=None):
    fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.matshow(matrix, cmap='viridis')

    # Add color bar
    fig.colorbar(cax)

    # Annotate each cell with the numeric value
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

    # Set custom tick marks if provided
    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90)  # Rotate labels if needed
    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
        
    plt.savefig(name, dpi=250, bbox_inches='tight')
    plt.clf()
    plt.close()

class Mapper():
    def __init__(self, config: dict, logger: Logger) -> None:
        # parse configuration
        self.config = config
        self.iterations = config["iterations"]
        self.new_submap_iterations = config["new_submap_iterations"]
        self.new_submap_points_num = config["new_submap_points_num"]
        self.new_submap_gradient_points_num = config["new_submap_gradient_points_num"]
        self.new_frame_sample_size = config["new_frame_sample_size"]
        self.new_points_radius = config["new_points_radius"]
        self.alpha_threshold = config["alpha_threshold"]
        self.pruning_threshold = config["pruning_threshold"]
        self.current_view_weight = config["current_view_weight"]

        
        self.isotropic_regularization = config["isotropic_regularization"]
        self.opacity_regularization = config["opacity_regularization"]

        self.optimize_affinity_features = config["affinity_features"]["optimize"]
        self.affinity_feature_optimization_method = config["affinity_features"]["method"]

        self.surface_regularization = config["surface_regularization"]
        self.normal_regularization = config["normal_regularization"]
        self.normal_smoothing = config["normal_smoothing"]
        self.scale_regularization = config["scale_regularization"]

        self.opt = OptimizationParams()

        self.logger = logger

        self.keyframes = []

    def compute_seeding_mask(self, keyframe: dict, gaussian_model: GaussianModel, new_submap: bool) -> np.ndarray:
        """
        Computes a binary mask to identify regions within a keyframe where new Gaussian models should be seeded.
        Seeding is based on color gradient for new submaps and alpha masks and depth error for existing submasks.

        Args:
            keyframe (dict): Keyframe dict containing color, depth, and render settings
            gaussian_model: The current submap
            new_submap (bool): A boolean indicating whether the seeding is occurring in current submap or a new submap
        Returns:
            np.ndarray: A binary mask of shpae (H, W) indicates regions suitable for seeding new 3D Gaussian models
        """
        seeding_mask = None
        if new_submap:
            color_image = keyframe["color"]
            seeding_mask = geometric_edge_mask(color_image, RGB=True)
        else:
            # TODO: check this again
            render_dict = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            alpha_mask = (render_dict["alpha"] < self.alpha_threshold)
            gt_depth_tensor = numpy_to_torch(keyframe["depth"], device="cuda")[None]
            depth_error = torch.abs(gt_depth_tensor - render_dict["depth"]) * (gt_depth_tensor > 0)
            normalized_depth_error = (depth_error / gt_depth_tensor) * (gt_depth_tensor > 0)
            # depth_error_mask = (render_dict["depth"] > gt_depth_tensor) * (depth_error > 40 * depth_error.median())
            depth_error_mask = (render_dict["depth"] > gt_depth_tensor) * (normalized_depth_error > 0.1)
            seeding_mask = alpha_mask | depth_error_mask
            seeding_mask = torch_to_numpy(seeding_mask[0])

        return seeding_mask
    
    def seed_new_gaussians(self, keyframe: dict, seeding_mask: np.ndarray, new_submap: bool) -> np.ndarray:
        """
        Seeds means for the new 3D Gaussian based on keyframe, a seeding mask, and a flag indicating whether this is a new submap.

        Args:
            keyframe: A dictionary with the current frame_id, gt_rgb, gt_depth, camera-to-world transform and image information.
                color: The ground truth color image as a numpy array with shape (H, W, 3).
                depth: The ground truth depth map as a numpy array with shape (H, W).
                K: The camera intrinsics matrix as a numpy array with shape (3, 3).
                T_c2w: The estimated camera-to-world transformation matrix as a numpy array with shape (4, 4).
            seeding_mask: A binary mask indicating where to seed new Gaussians, with shape (H, W).
            new_submap: Flag indicating whether the seeding is for a new submap (True) or an existing submap (False).
        Returns:
            np.ndarray: An array of 3D points where new Gaussians will be initialized, with shape (N, 6) (Last dimension containts xyzrgb values)
        """
        # pc_points = create_point_cloud(keyframe["color"], 1.005 * keyframe["depth"], keyframe["K"], keyframe["T_c2w"])
        pc_points = create_point_cloud(keyframe["color"], keyframe["depth"], keyframe["K"], keyframe["T_c2w"])

        flat_depth_mask = (keyframe["depth"] > 0.).flatten()
        valid_ids = np.flatnonzero(seeding_mask)
        if new_submap:
            if self.new_submap_points_num < 0:
                uniform_ids = np.arange(pc_points.shape[0])
            else:
                assert self.new_submap_points_num <= pc_points.shape[0] # don't sample more points than pixels
                uniform_ids = np.random.choice(pc_points.shape[0], self.new_submap_points_num, replace=False)
            gradient_ids = sample_pixels_based_on_gradient(keyframe["color"], self.new_submap_gradient_points_num)
            sample_ids = np.unique(np.concatenate((uniform_ids, gradient_ids, valid_ids)))
        else:
            if self.new_frame_sample_size < 0 or len(valid_ids) < self.new_frame_sample_size:
                sample_ids = valid_ids
            else:
                sample_ids = np.random.choice(valid_ids, self.new_frame_sample_size, replace=False)
        sample_ids = sample_ids[flat_depth_mask[sample_ids]]
        points = pc_points[sample_ids, :].astype(np.float32)
        print("--------------------")
        print("points: ", points.shape)
        print("--------------------")
        
        return points
    
    def grow_submap(self, keyframe: dict, seeded_points: np.ndarray, gaussian_model: GaussianModel, filter_cloud: bool) -> int:
        """
        Grows the current submap by adding new Gaussians to the GaussianModel object.

        Args:
            keyframe: A dictionary with the current frame_id, gt_rgb, gt_depth, camera-to-world transform and image information.
            seeded_points: An array of 3D points where new Gaussians will be initialized, with shape (N, 6) (xzyrgb).
            gaussian_model (GaussianModel): The current Gaussian model of the submap.
            filter_cloud: A boolean flag indicating whether to filter the point cloud for outliers/noise before adding to the submap.
        Returns:
            int: The number of new points added to the submap.
        """
        # get existing points in submap
        gaussian_points = gaussian_model.get_xyz()

        # determine subset of existing points within camera frustum
        camera_frustum_corners = compute_camera_frustum_corners(keyframe["depth"], keyframe["K"], keyframe["T_c2w"])
        existing_point_ids = compute_frustum_point_ids(gaussian_points, numpy_to_torch(camera_frustum_corners), device="cuda")

        # select new points to add to submap based on density of existing submaps points
        new_point_ids = select_new_point_ids(
            gaussian_points[existing_point_ids], 
            numpy_to_torch(seeded_points[:, :3]).contiguous(), # slice removes rgb values and takes points' xyz data only
            radius=self.new_points_radius,
            device="cuda"
        )
        new_point_ids = torch_to_numpy(new_point_ids)

        # add points
        if new_point_ids.shape[0] > 0:
            cloud_to_add = numpy_to_point_cloud(seeded_points[new_point_ids, :3], seeded_points[new_point_ids, 3:] / 255.0)
            if filter_cloud:
                cloud_to_add, _ = cloud_to_add.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
            gaussian_model.add_points(cloud_to_add)
        gaussian_model._features_dc.requires_grad = False
        gaussian_model._features_rest.requires_grad = False
        print("Gaussian model size", gaussian_model.get_size())
        
        return new_point_ids.shape[0]
    
    def optimize_submap(self, keyframes: list, gaussian_model: GaussianModel, iterations: int) -> dict:
        """
        Optimizes the submap by refining the parameters of the 3D Gaussian based on the observations
        from keyframes observing the submap.

        Args:
            keyframes: A list of tuples consisting of frame id and keyframe dictionary
            gaussian_model: An instance of the GaussianModel class representing the initial state
                of the Gaussian model to be optimized.
            iterations: The number of iterations to perform the optimization process. Defaults to 100.
        Returns:
            losses_dict: Dictionary with the optimization statistics
        """
        iteration = 0
        results_dict = {}
        num_keyframes = len(keyframes)
        
        # get view optimization distribution (how optimization iterations are distributed among keyframes)
        current_frame_iterations = self.current_view_weight * iterations
        view_distribution = keyframe_optimization_sampling_distribution(num_keyframes, iterations, current_frame_iterations)

        # check segmentation feature inclusion
        render_features = gaussian_model.include_point_features

        # optimize
        start_time = time.time()
        while iteration < iterations + 1:
            # initialize optimizer
            gaussian_model.set_optimizer_zero_grad()

            # sample view
            sampled_id = np.random.choice(np.arange(num_keyframes), p=view_distribution)
            keyframe = keyframes[sampled_id]
            frame_id = keyframe["frame_id"] # TODO: need a better solution for frame id management

            # render model
            render_dict = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            if render_features:
                render_dict["features"] = render_gaussian_model_features(gaussian_model, keyframe["render_settings"])["features"]
            
            visibility_filter = render_dict["radii"] > 0
            rendered_image, rendered_depth = render_dict["color"], render_dict["depth"]
            if render_features:
                rendered_features = render_dict["features"]

            # get ground truth
            gt_image = keyframe["color_torch"]
            gt_depth = keyframe["depth_torch"] # TODO: converted keyframe elements to tensor
            if render_features:
                gt_segmentation = keyframe["segmentation_torch"]
            gt_normals = keyframe["normals_torch"]

            # mask out invalid depth values
            mask = (gt_depth > 0) & (~torch.isnan(rendered_depth)).squeeze(0)

            # gaussian splatting losses
            total_loss = 0.0

            # compute depth loss
            depth_loss = l1_loss(rendered_depth[:, mask], gt_depth[mask])
            total_loss += depth_loss

            # compute color loss
            weight = self.opt.lambda_dssim
            pixelwise_color_loss = l1_loss(rendered_image[:, mask], gt_image[:, mask])
            ssim_loss = (1.0 - ssim(rendered_image, gt_image)) # TODO: check why mask isn't used here
            color_loss = (1.0 - weight) * pixelwise_color_loss + weight * ssim_loss
            total_loss += color_loss

            # compute isotropic regularization loss
            if self.isotropic_regularization:
                isotropic_regularization_loss = isotropic_loss(gaussian_model.get_scaling(), flat_gaussians=self.scale_regularization)
                total_loss += isotropic_regularization_loss

            # feature rendering loss terms
            # if render_features:
            if self.optimize_affinity_features:
                if self.affinity_feature_optimization_method == "contrastive":
                    segmentation_loss = contrastive_clustering_loss(
                        rendered_features.permute(1, 2, 0), 
                        gt_segmentation
                    )
                    
                    # rendered_feature_norm = render_dict["features"].norm(dim = 0, p=2).mean()
                    # rendered_feature_norm_reg = (1-rendered_feature_norm)**2

                    # spatial_loss = spatial_regularization_loss(gaussian_model.get_xyz(), gaussian_model.get_point_features())
                    # total_loss += segmentation_loss # + 1000*rendered_feature_norm_reg #+ spatial_loss #
                    # total_loss += segmentation_loss + 1000*rendered_feature_norm_reg
                if self.affinity_feature_optimization_method == "discriminative":
                    feature_dim = rendered_features.shape[0]
                    predicted_classes = gaussian_model.point_classifier(rendered_features.permute(1, 2, 0).reshape(-1, feature_dim))
                    target_classes = gt_segmentation.flatten().long()
                    segmentation_loss = pixel_classification_loss = F.cross_entropy(predicted_classes, target_classes)
                
                total_loss += segmentation_loss # + 1000*rendered_feature_norm_reg #+ spatial_loss #


            if self.opacity_regularization:
                visible_opacities = gaussian_model.get_opacity()[visibility_filter]
                total_loss +=  0.05 * opacity_entropy_regularization_loss(visible_opacities) # 0.05

            if self.surface_regularization:
                # sample control points
                gaussian_centers = gaussian_model.get_xyz()[visibility_filter]
                gaussian_rotations = gaussian_model.get_rotation()[visibility_filter]
                gaussian_scales = gaussian_model.get_scaling()[visibility_filter]
                control_points = sample_control_points(gaussian_centers, gaussian_rotations, gaussian_scales)

                # compute pixel coordinates of control points
                control_points_pixel_coordinates = world_points_to_pixel_coordinates(
                    control_points, 
                    numpy_to_torch(keyframe["K"], device=control_points.device), 
                    torch.linalg.inv(numpy_to_torch(keyframe["T_c2w"], device=control_points.device))
                )
                valid_heights = (control_points_pixel_coordinates[:, 0] >= 0) & (control_points_pixel_coordinates[:, 0] < keyframe["H"]) # check that these indices should be flipped ( 1 and 0  for pixel coords)
                valid_widths = (control_points_pixel_coordinates[:, 1] >= 0) & (control_points_pixel_coordinates[:, 1] < keyframe["W"])
                control_point_mask = valid_heights & valid_widths
            
                # get point cloud
                pointcloud = depth_image_to_pointcloud(
                    keyframe["depth_torch"], 
                    numpy_to_torch(keyframe["K"], device=keyframe["depth_torch"].device), 
                    numpy_to_torch(keyframe["T_c2w"], device=keyframe["depth_torch"].device)
                )
                pointcloud = pointcloud.reshape(keyframe["H"], keyframe["W"], 3)

                surface_points = pointcloud[control_points_pixel_coordinates[control_point_mask][:, 0], control_points_pixel_coordinates[control_point_mask][:, 1]]
                surface_loss = l1_loss(control_points[control_point_mask], surface_points)

                total_loss += surface_loss

            if self.normal_regularization:
                # oriented_normals = orient_gaussian_normals_toward_camera(
                #     gaussian_model.get_xyz().detach(),
                #     gaussian_model.get_normals(),
                #     numpy_to_torch(keyframe["T_c2w"][:3, 3], device="cuda").detach()
                # )
                rendered_normals = render_gaussian_model_normals(gaussian_model, keyframe["render_settings"]) #, override_colors=oriented_normals)
                # rendered_normals = F.normalize(rendered_normals, p=2, dim=0)
                normal_loss = l1_loss(rendered_normals, gt_normals.permute(2, 0, 1))
                total_loss += 0.1 * normal_loss

                if self.normal_smoothing:
                    # compute normal smoothing loss
                    normal_smoothness_loss = torch.mean(torch.abs(rendered_normals[:, :, 1:] - rendered_normals[:, :, :-1]))
                    normal_smoothness_loss += torch.mean(torch.abs(rendered_normals[:, 1:, :] - rendered_normals[:, :-1, :]))
                    total_loss += 0.01 * normal_smoothness_loss

            if self.scale_regularization:
                min_scales = torch.min(gaussian_model.get_scaling(), dim=1).values
                scale_loss = torch.abs(min_scales).sum()
                total_loss += 0.1 * scale_loss


            if self.isotropic_regularization:
                x = isotropic_regularization_loss.item()
            else:
                x = 0

            results_dict[frame_id] = {
                "depth_loss": depth_loss.item(),
                "color_loss": color_loss.item(),
                "isotropic_loss": x,
                "segmentation_loss": segmentation_loss.item() if render_features else None,
                # "rendered_feature_norm_reg": rendered_feature_norm_reg.item() if render_features else None,
                # "spatial_loss": spatial_loss.item() if render_features else None,
                "total_loss": total_loss.item()
            } # TODO: this isn't perfect logging, fix to keep track of all statistics

            # backpropagate
            total_loss.backward()

            with torch.no_grad():
                # check halfway and at end of optimization for points to remove based on opacity
                if iteration == iterations // 2 or iteration == iterations:
                    remove_mask = (gaussian_model.get_opacity() < self.pruning_threshold).squeeze()
                    gaussian_model.remove_points(remove_mask)

                # optimizer step
                if iteration < iterations:
                    gaussian_model.splat_optimizer.step()
                    if render_features:
                        gaussian_model.feature_optimizer.step()
                gaussian_model.set_optimizer_zero_grad()
            
            iteration += 1

            # print("REQUIRES GRAD: ", gaussian_model._features_dc.requires_grad)
            # print("features example: ", gaussian_model._features_dc[0])
            # if iteration == 100:
            #     breakpoint()
        
        
        # log optimization statistics
        torch.cuda.synchronize()
        optimization_time = time.time() - start_time
        results_dict["total_optimization_time"] = optimization_time
        results_dict["average_optimization_iteration_time"] = optimization_time / iterations
        results_dict["num_gaussians"] = gaussian_model.get_size()

        return results_dict

    def map(self, keyframe: dict, gaussian_model: GaussianModel, new_submap: bool) -> dict:
        """
        Mapping iteration that seeds new Gaussians, adds them to the submap, and then optimizes the submap.

        Args:
            keyframe_dict: A dictionary with the current frame_id, gt_rgb, gt_depth, camera-to-world transform and image information.
            gaussian_model (GaussianModel): The current Gaussian model of the submap
            is_new_submap (bool): A boolean flag indicating whether the current frame initiates a new submap
        Returns:
            opt_dict: Dictionary with statistics about the optimization process
        """
        # assemble keyframe
        render_segmentation_features = gaussian_model.include_point_features
        T_w2c = np.linalg.inv(keyframe["T_c2w"])
        keyframe["render_settings"] = get_render_settings(
            keyframe["H"],
            keyframe["W"],
            keyframe["K"],
            T_w2c,
        )
        color_transform = torchvision.transforms.ToTensor()
        keyframe["color_torch"] = color_transform(keyframe["color"]).cuda()
        keyframe["depth_torch"] = numpy_to_torch(keyframe["depth"], device="cuda") # TODO: converted keyframe elements to tensor
        if keyframe["segmentation"] is not None:
            keyframe["segmentation_torch"] = numpy_to_torch(keyframe["segmentation"], device="cuda")
        keyframe["normals_torch"] = numpy_to_torch(keyframe["normals"], device="cuda")

        # seed Gaussians
        seeding_mask = self.compute_seeding_mask(keyframe, gaussian_model, new_submap)
        seeded_points = self.seed_new_gaussians(keyframe, seeding_mask, new_submap)

        # add points to map
        filter_cloud = False # not new_submap # TODO: tune this
        new_pts_num = self.grow_submap(keyframe, seeded_points, gaussian_model, filter_cloud)

        # optimize submap
        max_iterations = self.new_submap_iterations if new_submap else self.iterations
        start_time = time.time()
        results_dict = self.optimize_submap([keyframe] + self.keyframes, gaussian_model, max_iterations)
        results_dict["new_submap"] = new_submap
        optimization_time = time.time() - start_time
        
        print("Optimization time: ", optimization_time)


        # append keyframe to list
        self.keyframes.append(keyframe)

        # visualizations and logging
        with torch.no_grad():
            render = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            render["segmentation"] = render_gaussian_model_features(gaussian_model, keyframe["render_settings"])["features"] # TODO: clean up
            rendered_image, rendered_depth = render["color"], render["depth"]
            
            oriented_normals = orient_gaussian_normals_toward_camera(
                gaussian_model.get_xyz().detach(),
                gaussian_model.get_normals(),
                numpy_to_torch(keyframe["T_c2w"][:3, 3], device="cuda").detach()
            )
            rendered_normals = render_gaussian_model_normals(gaussian_model, keyframe["render_settings"], override_colors=oriented_normals)



            psnr_value = psnr(rendered_image, keyframe["color_torch"]).mean().item()
            ssim_value = ssim(rendered_image, keyframe["color_torch"]).item()
            segmentation_loss = contrastive_clustering_loss(render["segmentation"].permute(1, 2, 0), keyframe["segmentation_torch"]) if render_segmentation_features else None
            # spatial_loss = spatial_regularization_loss(gaussian_model.get_xyz(), gaussian_model.get_point_features()) if render_segmentation_features else None

            rendered_feature_norm = render["segmentation"].norm(dim = 0, p=2).mean()
            rendered_feature_norm_reg = (1-rendered_feature_norm)**2

            results_dict["psnr_render"] = psnr_value
            results_dict["ssim_render"] = ssim_value
            print(f"PSNR this frame: {psnr_value}")
            print(f"SSIM this frame: {ssim_value}")
            print(f"Segmentation loss: {segmentation_loss}")
            print(f"Regularization loss: {rendered_feature_norm_reg}")
            self.logger.vis_mapping_iteration(
                keyframe["frame_id"], max_iterations,
                rendered_image.clone().detach().permute(1, 2, 0),
                rendered_depth.clone().detach().permute(1, 2, 0),
                keyframe["color_torch"].permute(1, 2, 0),
                keyframe["depth_torch"].unsqueeze(-1),
                rendered_normals.clone().detach().permute(1, 2, 0),
                keyframe["normals_torch"],
                seeding_mask=seeding_mask
            )

            ########### - BEGINNING OF SEGMENTATION VISUALIZATION - ###########
            if render_segmentation_features:
                if self.affinity_feature_optimization_method == "contrastive":
                    rendered_segmentation = render["segmentation"]

                    ###
                    from sklearn.decomposition import TruncatedSVD

                    feature_dim = rendered_segmentation.shape[0]
                    if feature_dim > 3:
                        # Reshape the features to a 2D array (height * width, feature_dim)
                        reshaped_features = torch_to_numpy(rendered_segmentation.permute(1,2,0)).reshape(-1, feature_dim)

                        # Perform SVD to reduce dimensions to 3
                        svd = TruncatedSVD(n_components=3)
                        projected_features = svd.fit_transform(reshaped_features)

                        # Reshape the projected features back to (height, width, 3)
                        projected_features_3d = torch.from_numpy(projected_features.reshape(keyframe["H"], keyframe["W"], 3))

                        self.logger.vis_segmentation_iteration(
                            keyframe["frame_id"], max_iterations,
                            projected_features_3d,
                            keyframe["segmentation_torch"],
                        )
                    else:
                    ###
                        self.logger.vis_segmentation_iteration(
                            keyframe["frame_id"], max_iterations,
                            rendered_segmentation.clone().detach().permute(1, 2, 0),
                            keyframe["segmentation_torch"],
                        )
                elif self.affinity_feature_optimization_method == "discriminative":
                    rendered_segmentation = render["segmentation"]
                    feature_dim = rendered_segmentation.shape[0]
                    predicted_classes = gaussian_model.point_classifier(rendered_segmentation.permute(1, 2, 0).reshape(-1, feature_dim))
                    predicted_classes = predicted_classes.argmax(dim=1).reshape(keyframe["H"], keyframe["W"])

                    self.logger.vis_segmentation_iteration_discriminative(
                        keyframe["frame_id"], max_iterations,
                        predicted_classes,
                        keyframe["segmentation_torch"],
                    )

        return results_dict

        
    



        
