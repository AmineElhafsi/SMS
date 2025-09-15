import torch

def psnr(image_1: torch.Tensor, image_2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        image_1: The first image.
        image_2: The second image.
    Returns:
        torch.Tensor: The computed PSNR.
    """
    mse = ((image_1 - image_2) ** 2).view(image_1.shape[0], -1).mean(1, keepdim=True)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr
