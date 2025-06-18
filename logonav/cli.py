"""CLI interface for logonav project."""

from collections import deque

import click
import cv2
import numpy as np
import torch
from PIL import Image

from logonav.inference import load_logonav_model, run_inference
from logonav.utils.transforms import to_numpy, transform_images_for_model
from logonav.utils.visualize import (
    backproject_pixel_to_bev,
    create_transform,
    draw_horizon_line,
    estimate_intrinsics,
    plot_3D_trajectory,
    project_world_to_image,
)


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
@click.option(
    "--camera-fov-x",
    type=float,
    default=120.0,
    help="Camera horizontal field of view in degrees.",
)
@click.option(
    "--camera-fov-y",
    type=float,
    default=90.0,
    help="Camera vertical field of view in degrees.",
)
@click.option(
    "--camera-height",
    type=float,
    default=1.5,
    help="Camera height from ground in meters.",
)
def process_video(
    video_path,
    model_path,
    device,
    output_path,
    goal_x,
    goal_y,
    goal_theta,
    camera_fov_x,
    camera_fov_y,
    camera_height,
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

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create camera matrices for 3D visualization
    intrinsic_matrix = estimate_intrinsics(
        fov_x=camera_fov_x,
        fov_y=camera_fov_y,
        height=height,
        width=width,
    )
    extrinsic_matrix = create_transform(
        x=0,
        y=0,
        z=0,
        roll=0,
        pitch=0,
        yaw=0,
    )

    # Create a video writer for the output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create a deque to store the last 6 frames
    frame_buffer = deque(maxlen=6)

    cv2.namedWindow("LogoNav")

    # Calculate default px py as the pixel coordinates of the
    # projection of the goal_x, goal_y on the camera plane
    goal_3d = np.array(
        [[goal_x, camera_height, goal_y]]
    )  # Convert BEV to 3D world coords
    default_goal_2d = project_world_to_image(
        goal_3d, intrinsic_matrix, extrinsic_matrix
    )
    print("goal_3d", goal_3d)
    print("default_goal_2d", default_goal_2d)
    goal_px, goal_py = int(default_goal_2d[0][0]), int(default_goal_2d[0][1])

    # Shared state for mouse callback
    mouse_state = {"goal_px": goal_px, "goal_py": goal_py}

    def onMouse(event, px, py, flags, param):
        # if event == cv2.EVENT_LBUTTONDOWN:
        #     # Update goal px py
        mouse_state["goal_px"] = px
        mouse_state["goal_py"] = py

    cv2.setMouseCallback("LogoNav", onMouse)

    goal_bev = (goal_x, goal_y)

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

            # Compute goal x y by taking goal px py
            # Take flat world assumption, assume px py are on the floor
            # Backproject to find the x y position in BEV space
            current_goal_px = mouse_state["goal_px"]
            current_goal_py = mouse_state["goal_py"]

            # Backproject pixel coordinates to world coordinates
            goal_bev = backproject_pixel_to_bev(
                current_goal_px,
                current_goal_py,
                intrinsic_matrix,
                extrinsic_matrix,
                camera_height,
                goal_bev,
            )
            current_goal_x, current_goal_y = goal_bev

            print("goal", (current_goal_px, current_goal_py), "->", goal_bev)

            # Goal pose [dx, dy, cos(theta), sin(theta)]
            goal_pose = torch.tensor(
                [
                    [
                        current_goal_x,
                        current_goal_y,
                        np.cos(goal_theta),
                        np.sin(goal_theta),
                    ]
                ]
            ).to(device, dtype=torch.float32)

            # Run inference
            waypoints = run_inference(model, obs_tensor, goal_pose)
            waypoints_np = to_numpy(waypoints)[0]

            # Visualize waypoints on the frame using new visualization tools
            vis_frame = visualize_waypoints_3d(
                frame,
                waypoints_np,
                intrinsic_matrix,
                extrinsic_matrix,
                camera_height,
                (0, 255, 0),
                True,
                5,
            )

            # Visualize goal pose in red
            vis_frame = visualize_waypoints_3d(
                vis_frame,
                np.array([[current_goal_x, current_goal_y]]),
                intrinsic_matrix,
                extrinsic_matrix,
                camera_height,
                (0, 0, 255),
                False,
                0,
            )

            vis_frame = draw_horizon_line(
                vis_frame,
                intrinsic_matrix,
                extrinsic_matrix,
                camera_height,
                color=(255, 255, 0),  # Yellow
                thickness=2,
            )

            cv2.imshow("LogoNav", vis_frame)
            key = cv2.waitKey(1)

            if key == ord("q"):
                exit()
            elif key == ord("d"):
                for _ in range(30 * 10):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1

            # Write the frame to the output video
            out.write(vis_frame)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames")

    # Release resources
    cap.release()
    out.release()
    print(f"Output video saved to {output_path}")


def visualize_waypoints_3d(
    frame,
    waypoints,
    intrinsic_matrix,
    extrinsic_matrix,
    camera_height,
    color,
    draw_line,
    interpolation_samples,
):
    """
    Visualize predicted waypoints using 3D projection tools.

    Args:
        frame: OpenCV frame
        waypoints: NumPy array of waypoints [N, 4] containing
            [x, y, cos(theta), sin(theta)]
        intrinsic_matrix: Camera intrinsic matrix
        extrinsic_matrix: Camera extrinsic matrix
        camera_height: Height of camera from ground in meters

    Returns:
        OpenCV frame with visualized waypoints
    """
    # Convert waypoints to 3D trajectory
    # Assume waypoint spacing of 0.25m (typical for LogoNav)
    metric_waypoint_spacing = 0.25

    trajectory_3D = [[0, camera_height, 0]]  # Start at camera position

    for waypoint in waypoints:
        # Scale waypoints from model output to metric coordinates
        wp_x = waypoint[0] * metric_waypoint_spacing
        wp_y = waypoint[1] * metric_waypoint_spacing

        # Add to trajectory (swap x/y to match coordinate system)
        trajectory_3D.append([wp_y, camera_height, wp_x])

    trajectory_3D = np.array(trajectory_3D)

    # Use the 3D visualization tool
    vis_frame = plot_3D_trajectory(
        frame_img=frame,
        trajectory_3D=trajectory_3D,
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=extrinsic_matrix,
        interpolation_samples=interpolation_samples,
        draw_line=draw_line,
        color=color,
    )

    # Compute speed and display
    if len(waypoints) > 1:
        # Calculate distance between first two waypoints
        wp1 = waypoints[0]
        wp2 = waypoints[1]

        # Distance in meters
        distance = (
            np.sqrt((wp2[0] - wp1[0]) ** 2 + (wp2[1] - wp1[1]) ** 2)
            * metric_waypoint_spacing
        )

        # Assume model predicts waypoints at 10 Hz (0.1 second intervals)
        time_interval = 0.1

        # Speed in m/s, then convert to km/h
        speed_ms = distance / time_interval
        speed_kmh = speed_ms * 3.6

        # Display speed or STOP
        if speed_kmh < 2.0:
            speed_text = "STOP"
        else:
            speed_text = f"{speed_kmh:4.1f}"

        # Add speed text to frame (top-left corner)
        cv2.putText(
            vis_frame,
            speed_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

    return vis_frame


def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m logonav` and `$ logonav `.

    This is the entry point for the LogoNav CLI.
    """
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
