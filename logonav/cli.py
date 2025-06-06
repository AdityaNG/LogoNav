"""CLI interface for logonav project."""

from collections import deque

import click
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from logonav.inference import load_logonav_model, run_inference
from logonav.utils.transforms import to_numpy, transform_images_for_model


@click.group()
def cli():
    """LogoNav: Visual Navigation with Goal Specifications."""
    pass


@cli.command()
@click.option(
    "--model-path",
    "-m",
    type=str,
    default=None,
    help="Path to model weights file. If not provided, will download pretrained weights.",  # noqa
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run inference on (cpu or cuda).",
)
def demo(model_path, device):
    """Run a simple demo of the LogoNav model."""
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load model with pretrained weights (will auto-download if needed)
    model = load_logonav_model(
        model_path=model_path,
        device=device,
        context_size=5,
        len_traj_pred=8,
        obs_encoding_size=1024,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
        pretrained=True,
    )

    # Create sample inputs (random tensors for demonstration)
    batch_size = 1
    image_size = (96, 96)
    context_size = 6

    # Random observation images
    random_obs = torch.rand(batch_size, 3 * context_size, *image_size).to(
        device
    )

    # Random goal pose [dx, dy, cos(theta), sin(theta)]
    random_goal = torch.tensor([[2.5, 1.0, 0.7071, 0.7071]]).to(device)

    # Run inference
    waypoints = run_inference(model, random_obs, random_goal)

    # Print results
    print(f"Predicted waypoints shape: {waypoints.shape}")
    print(f"Waypoints: {to_numpy(waypoints)}")

    return waypoints


@cli.command()
@click.option(
    "--video-path",
    "-v",
    required=True,
    type=str,
    help="Path to input video file.",
)
@click.option(
    "--model-path",
    "-m",
    type=str,
    default=None,
    help="Path to model weights file. If not provided, will download pretrained weights.",  # noqa
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run inference on (cpu or cuda).",
)
@click.option(
    "--output-path",
    "-o",
    type=str,
    default="output.mp4",
    help="Path to output video file.",
)
@click.option(
    "--goal-x",
    "-x",
    type=float,
    default=2.0,
    help="Goal X coordinate relative to current position.",
)
@click.option(
    "--goal-y",
    "-y",
    type=float,
    default=0.0,
    help="Goal Y coordinate relative to current position.",
)
@click.option(
    "--goal-theta",
    "-t",
    type=float,
    default=0.0,
    help="Goal orientation in radians.",
)
def process_video(
    video_path, model_path, device, output_path, goal_x, goal_y, goal_theta
):
    """Process a video with the LogoNav model and visualize waypoints."""
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load model with pretrained weights
    model = load_logonav_model(
        model_path=model_path,
        device=device,
        context_size=5,
        len_traj_pred=8,
        obs_encoding_size=1024,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
        pretrained=True,
    )

    # Goal pose [dx, dy, cos(theta), sin(theta)]
    goal_pose = torch.tensor(
        [[goal_x, goal_y, np.cos(goal_theta), np.sin(goal_theta)]]
    ).to(device)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a video writer for the output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create a deque to store the last 6 frames
    frame_buffer = deque(maxlen=6)

    # Process the video frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 96x96 for the model
        small_frame = cv2.resize(frame, (96, 96))
        # Convert BGR to RGB
        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_frame = Image.fromarray(small_frame_rgb)

        # Add to frame buffer
        frame_buffer.append(pil_frame)

        # Only process once we have enough frames
        if len(frame_buffer) == 6:
            # Convert frames to tensor
            obs_tensor = transform_images_for_model(list(frame_buffer)).to(
                device
            )

            # Run inference
            waypoints = run_inference(model, obs_tensor, goal_pose)
            waypoints_np = to_numpy(waypoints)[0]

            # Visualize waypoints on the frame
            vis_frame = visualize_waypoints(frame, waypoints_np, width, height)

            # Write the frame to the output video
            out.write(vis_frame)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames")

    # Release resources
    cap.release()
    out.release()
    print(f"Output video saved to {output_path}")


def visualize_waypoints(frame, waypoints, width, height):
    """
    Visualize predicted waypoints on the frame

    Args:
        frame: OpenCV frame
        waypoints: NumPy array of waypoints [N, 4]
        width: Frame width
        height: Frame height

    Returns:
        OpenCV frame with visualized waypoints
    """
    # Convert to PIL Image for easier drawing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_frame)

    # Scale factor from model coordinates to image coordinates
    # Assuming model coordinates are in meters and centered at the robot
    scale = min(width, height) / 10.0  # 10 meters covers the full frame

    # Center point (robot position)
    center_x = width // 2
    center_y = height // 2

    # Draw the waypoints
    for i, waypoint in enumerate(waypoints):
        # Extract position
        x, y = waypoint[0], waypoint[1]

        # Convert to image coordinates
        img_x = int(center_x + x * scale)
        img_y = int(center_y - y * scale)  # Negate y because image y is down

        # Draw a point for each waypoint
        radius = 5
        color = (255, 0, 0) if i == 0 else (0, 255, 0)
        draw.ellipse(
            (img_x - radius, img_y - radius, img_x + radius, img_y + radius),
            fill=color,
        )

        # Draw a line connecting waypoints
        if i > 0:
            prev_x, prev_y = waypoints[i - 1][0], waypoints[i - 1][1]
            prev_img_x = int(center_x + prev_x * scale)
            prev_img_y = int(center_y - prev_y * scale)
            draw.line(
                (prev_img_x, prev_img_y, img_x, img_y),
                fill=(0, 255, 255),
                width=2,
            )

    # Convert back to OpenCV format
    frame_with_waypoints = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
    return frame_with_waypoints


def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m logonav` and `$ logonav `.

    This is the entry point for the LogoNav CLI.
    """
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
