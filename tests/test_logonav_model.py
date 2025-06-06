"""Tests for the LogoNav model."""

import pytest
import torch
import numpy as np

from logonav.models.logonav_model import LogoNavModel
from logonav.inference import load_logonav_model, run_inference


@pytest.fixture
def device():
    """Fixture for device selection."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model_params():
    """Fixture for model parameters."""
    return {
        "context_size": 5,
        "len_traj_pred": 8,
        "learn_angle": True,
        "obs_encoding_size": 512,
        "mha_num_attention_heads": 2,
        "mha_num_attention_layers": 2,
        "mha_ff_dim_factor": 4,
    }


@pytest.fixture
def sample_inputs(device, model_params):
    """Fixture for sample model inputs."""
    batch_size = 2
    image_size = (96, 96)
    context_size = model_params["context_size"] + 1  # +1 for current frame

    # Random observation images
    random_obs = torch.rand(batch_size, 3 * context_size, *image_size).to(
        device
    )

    # Random goal pose [dx, dy, cos(theta), sin(theta)]
    random_goal = torch.tensor(
        [[2.5, 1.0, 0.7071, 0.7071], [1.0, 3.0, 1.0, 0.0]]
    ).to(device)

    return random_obs, random_goal


def test_model_initialization(model_params):
    """Test LogoNavModel initialization with different parameters."""
    model = LogoNavModel(**model_params)
    assert isinstance(model, LogoNavModel)

    # Test with different parameters
    model = LogoNavModel(
        context_size=3,
        len_traj_pred=5,
        learn_angle=False,
        obs_encoding_size=256,
    )
    assert model.context_size == 3
    assert model.len_trajectory_pred == 5
    assert (
        model.num_action_params == 2
    )  # Should be 2 when learn_angle is False


def test_model_forward_pass(device, model_params, sample_inputs):
    """Test the forward pass of the LogoNavModel."""
    model = LogoNavModel(**model_params).to(device)
    obs, goal = sample_inputs

    # Run forward pass
    output = model(obs, goal)

    # Check output shape
    batch_size = obs.shape[0]
    expected_shape = (
        batch_size,
        model_params["len_traj_pred"],
        4 if model_params["learn_angle"] else 2,
    )
    assert output.shape == expected_shape

    # Check if position deltas are accumulated
    # First waypoint should not be zero (in most cases)
    assert not torch.allclose(
        output[:, 0, :2], torch.zeros_like(output[:, 0, :2])
    )

    # Check if angles are normalized
    if model_params["learn_angle"]:
        # Check if the direction vectors are normalized
        angles_norm = torch.norm(output[:, :, 2:], dim=2)
        assert torch.allclose(
            angles_norm, torch.ones_like(angles_norm), atol=1e-6
        )


def test_load_logonav_model(device):
    """Test loading the LogoNav model."""
    model = load_logonav_model(
        model_path=None,  # Use default weights
        device=device,
        context_size=5,
        len_traj_pred=8,
        obs_encoding_size=1024,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
        pretrained=False,  # Don't download weights for testing
    )

    assert isinstance(model, LogoNavModel)
    assert model.context_size == 5
    assert model.len_trajectory_pred == 8


def test_run_inference(device, sample_inputs):
    """Test running inference with the model."""
    model = LogoNavModel(
        context_size=5,
        len_traj_pred=8,
        obs_encoding_size=512,
    ).to(device)

    obs, goal = sample_inputs

    # Run inference
    waypoints = run_inference(model, obs, goal)

    # Check output shape
    batch_size = obs.shape[0]
    expected_shape = (
        batch_size,
        8,
        4,
    )  # 8 waypoints, 4 params (x, y, cos, sin)
    assert waypoints.shape == expected_shape

    # Test if waypoints form a trajectory (each point builds on previous)
    # The distance between consecutive waypoints should be reasonable
    waypoints_np = waypoints.detach().cpu().numpy()
    for b in range(batch_size):
        for i in range(1, waypoints_np.shape[1]):
            prev_pos = waypoints_np[b, i - 1, :2]
            curr_pos = waypoints_np[b, i, :2]
            # Distance between consecutive waypoints shouldn't be too large
            distance = np.linalg.norm(curr_pos - prev_pos)
            assert distance < 5.0, f"Waypoint distance too large: {distance}"
