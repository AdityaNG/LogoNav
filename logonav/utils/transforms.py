"""Transform utilities for LogoNav."""

from typing import List

import torch
from PIL import Image
from torchvision import transforms


def transform_images_for_model(pil_imgs: List[Image.Image]) -> torch.Tensor:
    """
    Transforms a list of PIL images to a torch tensor with normalization.

    Args:
        pil_imgs: List of PIL images

    Returns:
        torch.Tensor: Transformed images concatenated along the
            channel dimension
    """
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]

    transformed_imgs = []
    for pil_img in pil_imgs:
        transformed_img = transform_type(pil_img)
        transformed_img = torch.unsqueeze(transformed_img, 0)
        transformed_imgs.append(transformed_img)

    return torch.cat(transformed_imgs, dim=1)


def clip_angle(angle):
    """
    Clip angle to [-pi, pi]

    Args:
        angle: Angle in radians

    Returns:
        float: Angle clipped to [-pi, pi]
    """
    import numpy as np

    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def to_numpy(tensor):
    """
    Convert a PyTorch tensor to numpy array

    Args:
        tensor: PyTorch tensor

    Returns:
        numpy.ndarray: NumPy array
    """
    return tensor.cpu().detach().numpy()
