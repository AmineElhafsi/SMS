import math

import torch
import torch.nn.functional as F

# metrics and loss functions:
def contrastive_clustering_loss(feature_map, instance_segmentation):
    """
    Compute the contrastive clustering loss following https://arxiv.org/pdf/2311.11666 / https://arxiv.org/pdf/2404.12784v1.

    Args:
        feature_map: The feature map of shape (H, W, C).
        instance_segmentation: The instance segmentation of shape (H, W).
    Returns:
        torch.Tensor: The contrastive clustering loss.

    """
    # normalize features
    normalized_feature_map = F.normalize(feature_map, p=2, dim=2)
    
    # determine unique instances, obtain their masks and compute respective mean features
    instance_ids = torch.unique(instance_segmentation)
    instance_masks = torch.stack([instance_segmentation == id for id in instance_ids])
    mean_features = torch.stack([normalized_feature_map[mask].mean(dim=0) for mask in instance_masks])
    mean_features = F.normalize(mean_features, p=2, dim=1)

    # compute temperatures
    phi = torch.zeros(len(instance_ids)).to("cuda:0")
    for i, mask in enumerate(instance_masks):
        Np = mask.sum()
        phi[i] = torch.norm(normalized_feature_map[mask] - mean_features[i], dim=1).sum() / (Np * torch.log(Np + 100))
        
    phi = torch.clip(phi * 10, min=.1, max=1.0)
    phi = phi.detach()

    # compute dot products
    dot_products = torch.einsum('ijk,lk->ijl', normalized_feature_map, mean_features) # H, W, num_classes # --> dot_products[i,j,l] = f_(i, j) dot f_bar^(class_l)

    instance_losses = torch.zeros(len(instance_ids)).to("cuda:0")
    distance_losses = torch.zeros(len(instance_ids)).to("cuda:0")
    for id, mask in enumerate(instance_masks):
        numerator = torch.exp(dot_products[mask][:, id] / phi[id])
        denominator = torch.exp(dot_products[mask] / phi[None, :]).sum(dim=1) + 1e-6
        instance_losses[id] = -torch.log(numerator / denominator).sum()
        # distance_losses[id] = torch.abs(torch.norm(feature_map[mask], dim=1) - 1).sum()

    # print("distance_losses: ", distance_losses.sum())
    # print("instance_losses: ", instance_losses.sum())

    
    loss = instance_losses.sum() / len(instance_ids) # + distance_losses.sum() / len(instance_ids)

    if loss.isnan():
        print("Segmentation loss is nan")
        breakpoint()

    return loss


def isotropic_loss(scaling: torch.Tensor, flat_gaussians: bool) -> torch.Tensor:
    """
    Computes the isotropic loss to reduce the emergence of elongated 3D Gaussians.

    Args:
        scaling: The scaling tensors for the 3D Gaussians of shape(N, 3).
    Returns:
        torch.Tensor: The computed isotropic loss.
    """
    if flat_gaussians:
        k = 2
        scaling, indices = torch.topk(scaling, k, dim=1)

    mean_scaling = scaling.mean(dim=1, keepdim=True)
    isotropic_diff = torch.abs(scaling - mean_scaling * torch.ones_like(scaling))
    return isotropic_diff.mean()


def l1_loss(prediction: torch.Tensor, target: torch.Tensor, aggregation_method="mean") -> torch.Tensor:
    """
    Computes the L1 loss between a prediction and a target. Optionally specify an aggregation method.

    Args:
        prediction: The predicted tensor.
        target: The ground truth tensor.
        aggregation_method: The aggregation method to be used. Defaults to "mean".
    Returns:
        torch.Tensor: The computed L1 loss.
    """
    l1_loss = torch.abs(prediction - target)
    if aggregation_method == "mean":
        return l1_loss.mean()
    elif aggregation_method == "sum":
        return l1_loss.sum()
    elif aggregation_method == "none":
        return l1_loss
    else:
        raise ValueError("Invalid aggregation method.")
    

def opacity_entropy_regularization_loss(opacities: torch.Tensor) -> torch.Tensor:
    """
    Computes the entropy regularization loss for the Gaussian opacities.

    Args:
        p: The probability distribution.
    Returns:
        torch.Tensor: The computed entropy loss.
    """
    loss = (-opacities * torch.log(opacities + 1e-10) - (1 - opacities) * torch.log(1 - opacities + 1e-10)).mean()
    return loss
    

def spatial_regularization_loss(xyz, features, k=64, max_points=50000, sample_size=10000):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using top-k neighbors.
    """
    assert xyz.size(0) == features.size(0)

    # don't modify the positions
    xyz = xyz.detach().clone()

    # normalize features
    features = F.normalize(features, p=2, dim=1)
    
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        xyz = xyz[indices]


    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_xyz = xyz[indices]
    sample_features = features[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_xyz, sample_xyz) + torch.diag(torch.inf * torch.ones(sample_xyz.shape[0], device=xyz.device)) # Compute pairwise distances
    _, near_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances
    _, far_indices_tensor = dists.topk(k, largest=True)  # Get top-k largest distances

    # Gather the neighbors
    near_neighbors = features[near_indices_tensor]
    far_neighbors = features[far_indices_tensor]

    # Compute the dot products between each feature and its k nearest neighbors
    near_dot_products = torch.einsum('ijk,ik->ij', near_neighbors, sample_features)
    far_dot_products = torch.einsum('ijk,ik->ij', far_neighbors, sample_features)

    loss = 10000 * (1 - near_dot_products).mean() + 10000 * far_dot_products.mean()
    return loss


def ssim(image_1: torch.Tensor, image_2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Args:
        image_1: The first image.
        image_2: The second image.
        window_size: The size of the window for SSIM computation. Defaults to 11.
        size_average: Flag that averages the SSIM over all image pixels if True.
    Returns:
        torch.Tensor: The computed SSIM.
    """
    num_channels = image_1.size(-3) 
    
    # create 2D Gaussian kernel
    sigma = 1.5 # TODO: I don't like that this is hardcoded here
    gaussian = torch.Tensor(
        [math.exp(-1/2 * (x - window_size // 2) ** 2 / float(sigma ** 2)) for x in range(window_size)]
    )
    kernel_1d = (gaussian / gaussian.sum()).unsqueeze(1)
    kernel_2d = kernel_1d.mm(kernel_1d.t()).unsqueeze(0).unsqueeze(0)
    window = kernel_2d.expand(num_channels, 1, window_size, window_size).contiguous() # TODO: check if torch.autograd.Variable is needed

    # ensure correct device and type
    if image_1.is_cuda:
        window = window.cuda(image_1.get_device())
    else:
        raise ValueError("SSIM computation requires CUDA.")
    window = window.type_as(image_1)

    # compute ssim
    mu1 = F.conv2d(image_1, window, padding=window_size//2, groups=num_channels)
    mu2 = F.conv2d(image_2, window, padding=window_size//2, groups=num_channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image_1 * image_1, window, padding=window_size//2, groups=num_channels) - mu1_sq
    sigma2_sq = F.conv2d(image_2 * image_2, window, padding=window_size//2, groups=num_channels) - mu2_sq
    sigma12 = F.conv2d(image_1 * image_2, window, padding=window_size//2, groups=num_channels) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_value = ssim_numerator / ssim_denominator

    if size_average:
        return ssim_value.mean()
    else:
        return ssim_value.mean(1).mean(1).mean(1)