# LogoNav

[![codecov](https://codecov.io/gh/AdityaNG/LogoNav/branch/main/graph/badge.svg?token=LogoNav_token_here)](https://codecov.io/gh/AdityaNG/LogoNav)
[![CI](https://github.com/AdityaNG/LogoNav/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/LogoNav/actions/workflows/main.yml)

LogoNav: Long-range Goal Pose conditioned Navigation policy

## Install it from PyPI

```bash
pip install logonav
```

## Usage

### Basic Model Loading and Inference

```python
import torch
from PIL import Image
from logonav import load_logonav_model, transform_images_for_model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model (will auto-download weights if needed)
model = load_logonav_model(
    device=device,
    pretrained=True
)

# Prepare input observations (assume we have 6 frames as PIL Images)
frames = [Image.open(f"frame_{i}.jpg").resize((96, 96)) for i in range(6)]
obs_tensor = transform_images_for_model(frames).to(device)

# Set goal pose [dx, dy, cos(theta), sin(theta)]
goal_pose = torch.tensor([[2.0, 0.5, 0.707, 0.707]]).to(device)

# Run inference
with torch.no_grad():
    waypoints = model(obs_tensor, goal_pose)

print(f"Predicted waypoints shape: {waypoints.shape}")
# Outputs: [batch_size, 8, 4] where each waypoint is [x, y, cos(theta), sin(theta)]
```

### Command Line Interface

The package includes a command-line interface for quick demos and video processing:

#### Run a simple demo

```bash
$ python -m logonav demo
# or
$ logonav demo
```

#### Process a video file and visualize waypoints

```bash
$ python -m logonav process-video --video-path input.mp4 --output-path output.mp4 --goal-x 2.0 --goal-y 0.5
# or
$ logonav process-video -v input.mp4 -o output.mp4 -x 2.0 -y 0.5
```

### Advanced Usage: Custom Configuration

```python
import torch
from logonav import load_logonav_model

# Load model with custom parameters
model = load_logonav_model(
    model_path="path/to/custom/weights.pth",  # Optional: use custom weights
    device=torch.device("cuda"),
    context_size=5,                          # Number of context frames
    len_traj_pred=8,                         # Number of waypoints to predict
    obs_encoding_size=1024,                  # Size of observation encoding
    mha_num_attention_heads=4,               # Number of attention heads
    mha_num_attention_layers=4,              # Number of transformer layers
    pretrained=False                         # Don't download pretrained weights
)
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
