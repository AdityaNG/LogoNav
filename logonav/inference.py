"""Inference utilities for LogoNav model."""

import os
from typing import Optional

import torch

from logonav.models.logonav_model import LogoNavModel
from logonav.utils.download import download_and_extract_model_weights


def load_logonav_model(
    model_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    context_size: int = 5,
    len_traj_pred: int = 8,
    obs_encoding_size: int = 1024,
    mha_num_attention_heads: int = 4,
    mha_num_attention_layers: int = 4,
    mha_ff_dim_factor: int = 4,
    pretrained: bool = False,
    cache_dir: Optional[str] = None,
) -> LogoNavModel:
    """
    Load the LogoNav model from a checkpoint file

    Args:
        model_path: Path to the model weights file. If None and
            pretrained=True, will download weights
        device: PyTorch device to load the model on
        context_size: Number of context frames
        len_traj_pred: Number of waypoints to predict
        obs_encoding_size: Size of observation encoding
        mha_num_attention_heads: Number of attention heads
        mha_num_attention_layers: Number of transformer layers
        mha_ff_dim_factor: Factor for feedforward network dimension
        pretrained: If True and model_path is None, will download
            pretrained weights
        cache_dir: Directory to store downloaded weights if pretrained=True

    Returns:
        LogoNavModel: Loaded model
    """
    # Initialize model
    model = LogoNavModel(
        context_size=context_size,
        len_traj_pred=len_traj_pred,
        learn_angle=True,
        obs_encoder="efficientnet-b0",
        obs_encoding_size=obs_encoding_size,
        late_fusion=False,
        mha_num_attention_heads=mha_num_attention_heads,
        mha_num_attention_layers=mha_num_attention_layers,
        mha_ff_dim_factor=mha_ff_dim_factor,
    )

    # Download weights if pretrained and no model_path specified
    if model_path is None and pretrained:
        model_path = download_and_extract_model_weights(cache_dir)

    # Load weights
    if model_path is not None and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        print(f"Successfully loaded model from {model_path}")
    elif model_path is not None:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    else:
        print("No model weights loaded. Using randomly initialized weights.")

    model.to(device)
    model.eval()
    return model


def run_inference(
    model: LogoNavModel, obs_tensor: torch.Tensor, goal_pose: torch.Tensor
) -> torch.Tensor:
    """
    Run inference with the LogoNav model

    Args:
        model: LogoNav model
        obs_tensor: Observation tensor [B, 3*context_size, H, W]
        goal_pose: Goal pose tensor [B, 4] (dx, dy, cos, sin)

    Returns:
        torch.Tensor: Predicted waypoints
    """
    with torch.no_grad():
        return model(obs_tensor, goal_pose)
