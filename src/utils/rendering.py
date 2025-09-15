import numpy as np
import torch

from gaussian_rasterizer import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_feature import GaussianRasterizer as GaussianFeatureRasterizer

from src.splatting.gaussian_model import GaussianModel


def get_render_settings(
    H: int, 
    W: int, 
    K: np.ndarray, 
    T_w2c: np.ndarray, 
    near: float = 0.01, 
    far: float = 100.0, 
    sh_degree: int = 0,
):
    """
    Constructs and returns a GaussianRasterizationSettings object for rendering,
    configured with given camera parameters.

    Args:
        H (int): The height of the image.
        W (int): The width of the image.
        K (array): Intrinsic camera matrix (3 x 3).
        w2c (array): World to camera transformation matrix.
        near (float, optional): The near plane for the camera. Defaults to 0.01.
        far (float, optional): The far plane for the camera. Defaults to 100.
    Returns:
        GaussianRasterizationSettings: Configured settings for Gaussian rasterization.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    T_w2c = torch.tensor(T_w2c).cuda().float()
    camera_center = torch.inverse(T_w2c)[:3, 3]
    tanfovx = W / (2.0 * fx)
    tanfovy = H / (2.0 * fy)

    view_matrix = T_w2c.transpose(0, 1)
    opengl_proj_matrix = torch.tensor(
        [
            [2 * fx / W, 0.0, -(W - 2 * cx) / W, 0.0],
            [0.0, 2 * fy / H, -(H - 2 * cy) / H, 0.0],
            [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device="cuda").float().transpose(0, 1)
    full_proj_matrix = view_matrix.unsqueeze(0).bmm(opengl_proj_matrix.unsqueeze(0)).squeeze(0)

    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx.astype(np.float32),
        tanfovy=tanfovy.astype(np.float32),
        bg=torch.zeros(3, device="cuda").float(),
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=full_proj_matrix,
        sh_degree=sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=False,
    )

    return settings


def render_gaussian_model(
    gaussian_model: GaussianModel,
    render_settings: GaussianRasterizationSettings,
    override_means_3d=None,
    override_means_2d=None,
    override_scales=None,
    override_rotations=None,
    override_opacities=None,
    override_colors=None,
):
    """
    Renders a Gaussian model using the given settings. Optionally override various model parameters.

    Args:
        gaussian_model: A Gaussian model object that provides methods to get
            various properties like xyz coordinates, opacity, features, etc.
        render_settings: Configuration settings for the GaussianRasterizer.
        override_means_3d (Optional): If provided, these values will override
            the 3D mean values from the Gaussian model.
        override_means_2d (Optional): If provided, these values will override
            the 2D mean values. Defaults to zeros if not provided.
        override_scales (Optional): If provided, these values will override the
            scale values from the Gaussian model.
        override_rotations (Optional): If provided, these values will override
            the rotation values from the Gaussian model.
        override_opacities (Optional): If provided, these values will override
            the opacity values from the Gaussian model.
        override_colors (Optional): If provided, these values will override the
            color values from the Gaussian model.
    Returns:
        dict: A dictionary containing the rendered color, depth, radii, 2D means, and alphas
        of the Gaussian model. The keys of this dictionary are 'color', 'depth',
        'radii', 'means2D', and 'alpha', each mapping to their respective rendered values.
    """
    renderer = GaussianRasterizer(raster_settings=render_settings)

    if override_means_3d is None:
        means3D = gaussian_model.get_xyz()
    else:
        means3D = override_means_3d

    if override_means_2d is None:
        means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
        means2D.retain_grad()
    else:
        means2D = override_means_2d

    if override_scales is None:
        scales = gaussian_model.get_scaling()
    else:
        scales = override_scales

    if override_rotations is None:
        rotations = gaussian_model.get_rotation()
    else:
        rotations = override_rotations

    if override_opacities is None:
        opacities = gaussian_model.get_opacity()
        # opacities = gaussian_model.get_opacity_point_features().detach()
    else:
        opacities = override_opacities

    shs, colors_precomp = None, None
    if override_colors is None:
        shs = gaussian_model.get_features()
    else:
        colors_precomp = override_colors

    # # TODO: Implement auxiliary features
    # if override_auxiliary_features is None:
    #     auxiliary_features = gaussian_model.get_segmentation_features()
    # else:
    #     auxiliary_features = override_auxiliary_features

    render_args = {
        "means3D": means3D,
        "means2D": means2D,
        "opacities": opacities,
        "colors_precomp": colors_precomp,
        "shs": shs,
        # "auxiliary_features_precomp": auxiliary_features,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": None,
    }
    # color, segmentation, depth, alpha, radii = renderer(**render_args)
    color, depth, alpha, radii = renderer(**render_args)

    # return {"color": color, "segmentation": segmentation, "depth": depth, "radii": radii, "means2D": means2D, "alpha": alpha}        
    return {"color": color, "depth": depth, "radii": radii, "means2D": means2D, "alpha": alpha}        

def render_gaussian_model_features(
    gaussian_model: GaussianModel,
    render_settings: GaussianRasterizationSettings,
    override_means_3d=None,
    override_means_2d=None,
    override_scales=None,
    override_rotations=None,
    override_opacities=None,
    override_colors=None,
):
    """
    Renders a Gaussian model using the given settings. Optionally override various model parameters.

    Args:
        gaussian_model: A Gaussian model object that provides methods to get
            various properties like xyz coordinates, opacity, features, etc.
        render_settings: Configuration settings for the GaussianRasterizer.
        override_means_3d (Optional): If provided, these values will override
            the 3D mean values from the Gaussian model.
        override_means_2d (Optional): If provided, these values will override
            the 2D mean values. Defaults to zeros if not provided.
        override_scales (Optional): If provided, these values will override the
            scale values from the Gaussian model.
        override_rotations (Optional): If provided, these values will override
            the rotation values from the Gaussian model.
        override_opacities (Optional): If provided, these values will override
            the opacity values from the Gaussian model.
        override_colors (Optional): If provided, these values will override the
            color values from the Gaussian model.
    Returns:
        dict: A dictionary containing the rendered color, depth, radii, 2D means, and alphas
        of the Gaussian model. The keys of this dictionary are 'color', 'depth',
        'radii', 'means2D', and 'alpha', each mapping to their respective rendered values.
    """
    renderer = GaussianFeatureRasterizer(raster_settings=render_settings)

    if override_means_3d is None:
        means3D = gaussian_model.get_xyz()
    else:
        means3D = override_means_3d

    if override_means_2d is None:
        means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
        means2D.retain_grad()
    else:
        means2D = override_means_2d

    if override_scales is None:
        scales = gaussian_model.get_scaling()
    else:
        scales = override_scales

    if override_rotations is None:
        rotations = gaussian_model.get_rotation()
    else:
        rotations = override_rotations

    if override_opacities is None:
        opacities = gaussian_model.get_opacity_point_features().detach()
    else:
        opacities = override_opacities

    shs, colors_precomp = None, None
    if override_colors is None:
        colors_precomp = gaussian_model.get_point_features()
    else:
        colors_precomp = override_colors

    # # TODO: Implement auxiliary features
    # if override_auxiliary_features is None:
    #     auxiliary_features = gaussian_model.get_segmentation_features()
    # else:
    #     auxiliary_features = override_auxiliary_features

    render_args = {
        "means3D": means3D.detach(),
        "means2D": means2D.detach(),
        "opacities": opacities.detach(),
        "colors_precomp": colors_precomp,
        "shs": shs,
        # "auxiliary_features_precomp": auxiliary_features,
        "scales": scales.detach(),
        "rotations": rotations.detach(),
        "cov3D_precomp": None,
    }
    # color, segmentation, depth, alpha, radii = renderer(**render_args)
    features, radii = renderer(**render_args)

    # return {"color": color, "segmentation": segmentation, "depth": depth, "radii": radii, "means2D": means2D, "alpha": alpha}        
    return {"features": features, "radii": radii} 

def render_gaussian_model_normals(
    gaussian_model: GaussianModel,
    render_settings: GaussianRasterizationSettings,
    override_means_3d=None,
    override_means_2d=None,
    override_scales=None,
    override_rotations=None,
    override_opacities=None,
    override_colors=None,
):
    """
    Renders a Gaussian model using the given settings. Optionally override various model parameters.

    Args:
        gaussian_model: A Gaussian model object that provides methods to get
            various properties like xyz coordinates, opacity, features, etc.
        render_settings: Configuration settings for the GaussianRasterizer.
        override_means_3d (Optional): If provided, these values will override
            the 3D mean values from the Gaussian model.
        override_means_2d (Optional): If provided, these values will override
            the 2D mean values. Defaults to zeros if not provided.
        override_scales (Optional): If provided, these values will override the
            scale values from the Gaussian model.
        override_rotations (Optional): If provided, these values will override
            the rotation values from the Gaussian model.
        override_opacities (Optional): If provided, these values will override
            the opacity values from the Gaussian model.
        override_colors (Optional): If provided, these values will override the
            color values from the Gaussian model.
    Returns:
        dict: A dictionary containing the rendered color, depth, radii, 2D means, and alphas
        of the Gaussian model. The keys of this dictionary are 'color', 'depth',
        'radii', 'means2D', and 'alpha', each mapping to their respective rendered values.
    """
    renderer = GaussianRasterizer(raster_settings=render_settings)

    if override_means_3d is None:
        means3D = gaussian_model.get_xyz()
    else:
        means3D = override_means_3d

    if override_means_2d is None:
        means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
        means2D.retain_grad()
    else:
        means2D = override_means_2d

    if override_scales is None:
        scales = gaussian_model.get_scaling()
    else:
        scales = override_scales

    if override_rotations is None:
        rotations = gaussian_model.get_rotation()
    else:
        rotations = override_rotations

    if override_opacities is None:
        opacities = gaussian_model.get_opacity()
    else:
        opacities = override_opacities

    shs, colors_precomp = None, None
    if override_colors is None:
        colors_precomp = gaussian_model.get_normals()
    else:
        colors_precomp = override_colors

    # assemble arg dict for rendering
    render_args = {
        "means3D": means3D,
        "means2D": means2D,
        "opacities": opacities,
        "colors_precomp": colors_precomp,
        "shs": shs,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": None,
    }

    # render
    # rendered_normals, _ = renderer(**render_args)
    rendered_normals, _, _, _ = renderer(**render_args)

    return rendered_normals