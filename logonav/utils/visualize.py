"""
Plotting utils

Inspiration
trajectory plotting utils: https://github.com/AdityaNG/general-navigation/blob/306fd4eed07a54b0fbc5b6df0ecd1dc78f8ba497/general_navigation/models/model_utils.py  # noqa
"""

from typing import Optional, Tuple

import cv2
import numpy as np


###############################################################################
# Camera Utils
def estimate_intrinsics(
    fov_x: float,  # degrees
    fov_y: float,  # degrees
    height: int,  # pixels
    width: int,  # pixels
) -> np.ndarray:
    """
    The intrinsic matrix can be extimated from the FOV and image dimensions

    :param fov_x: FOV on x axis in degrees
    :type fov_x: float
    :param fov_y: FOV on y axis in degrees
    :type fov_y: float
    :param height: Height in pixels
    :type height: int
    :param width: Width in pixels
    :type width: int
    :returns: (3,3) intrinsic matrix
    """
    fov_x = np.deg2rad(fov_x)
    fov_y = np.deg2rad(fov_y)

    if fov_x == 0.0 or fov_y == 0.0:
        raise ZeroDivisionError("fov can't be zero")

    c_x = width / 2.0
    c_y = height / 2.0
    f_x = c_x / np.tan(fov_x / 2.0)
    f_y = c_y / np.tan(fov_y / 2.0)

    intrinsic_matrix = np.array(
        [
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1],
        ],
        dtype=np.float16,
    )

    return intrinsic_matrix


def apply_transform(
    points_3D: np.ndarray,
    transform_matrix: np.ndarray,
) -> np.ndarray:
    """
    Takes an (N,3) list of 3D points
    transform_matrix is (4,4)
    Returns an (N,3) list of 2D points on the camera plane

    :param points_3D: (N,3) list of 3D points
    :type points_3D: np.ndarray
    :param transform_matrix: (4,4) matrix
    :type transform_matrix: np.ndarray
    :returns: (N,3) list of 2D points on the camera plane
    """

    if len(transform_matrix.shape) != 2 or transform_matrix.shape != (4, 4):
        raise ValueError(
            "transform_matrix expected shape (4, 4), "
            + f"got {transform_matrix.shape}"
        )

    if len(points_3D.shape) != 2 or points_3D.shape[1] != 3:
        raise ValueError(
            "points_3D expected shape (N, 3), " + f"got {points_3D.shape}"
        )

    # points_3D is (N, 3)
    # points_3D_homo is (N, 4)
    # extrinsic_matrix is (4, 4)
    points_3D_homo = np.array(
        [
            points_3D[:, 0],
            points_3D[:, 1],
            points_3D[:, 2],
            np.ones_like(points_3D[:, 0]),
        ]
    ).T

    # points_3D_homo_transformed is (N, 4)
    points_3D_homo_transformed = (transform_matrix @ points_3D_homo.T).T

    return points_3D_homo_transformed[:, :3]


def project_world_cam_to_image(
    points_3D_cam: np.ndarray, intrinsic_matrix: np.ndarray
) -> np.ndarray:
    """
    Project 3D points in camera coordinates to 2D image coordinates.

    :param points_3D_cam: (N,3) list of 3D points in camera coordinates
    :type points_3D_cam: np.ndarray
    :param intrinsic_matrix: (3,3) intrinsic matrix
    :type intrinsic_matrix: np.ndarray
    :returns: (N,2) list of 2D points on the image plane
    """
    if len(intrinsic_matrix.shape) != 2 or intrinsic_matrix.shape != (3, 3):
        raise ValueError(
            "intrinsic_matrix expected shape (3, 3), "
            + f"got {intrinsic_matrix.shape}"
        )

    if len(points_3D_cam.shape) != 2 or points_3D_cam.shape[1] != 3:
        raise ValueError(
            "points_3D_cam expected shape (N, 3), "
            + f"got {points_3D_cam.shape}"
        )

    intrinsic_matrix_homo = np.eye(4)
    intrinsic_matrix_homo[:3, :3] = intrinsic_matrix

    points_3D_homo = np.array(
        [
            points_3D_cam[:, 0],
            points_3D_cam[:, 1],
            points_3D_cam[:, 2],
            np.ones_like(points_3D_cam[:, 0]),
        ]
    ).T
    points_2D_homo = (intrinsic_matrix_homo @ points_3D_homo.T).T
    points_2D = np.array(
        [
            points_2D_homo[:, 0] / points_2D_homo[:, 2],
            points_2D_homo[:, 1] / points_2D_homo[:, 2],
        ]
    ).T

    return points_2D


def project_world_to_image(
    points_3D: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
) -> np.ndarray:
    """
    Takes an (N,3) list of 3D points
    intrinsic_matrix is (3,3)
    Returns an (N,3) list of 2D points on the camera plane

    :param points_3D: (N,3) list of 3D points
    :type points_3D: np.ndarray
    :param intrinsic_matrix: (3,3) intrinsics
    :type intrinsic_matrix: np.ndarray
    :param extrinsic_matrix: offsets to adjust the trajectory by
    :type extrinsic_matrix: np.ndarray
    :returns: (N,2) list of 2D points on the camera plane
    """
    points_3D_cam = apply_transform(points_3D, extrinsic_matrix)
    points_2D = project_world_cam_to_image(points_3D_cam, intrinsic_matrix)
    return points_2D


def project_image_to_world(
    image: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    subsample: int = 1,
    mask: Optional[np.ndarray] = None,
    bounds: float = float("inf"),  # meters
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes depth and the image as input and produces 3D pointcloud with color occupancy
    https://github.com/AdityaNG/socc_plotter/blob/2fda52641d2353e56b4f8fd280e789105981ff1b/socc_plotter/socc.py#L10-L77  # noqa

    Args:
        image (np.ndarray): (HxWx3) uint8
        depth (np.ndarray): (HxW) float32
        intrinsics (Optional[np):ndarray]): 3x3
        subsample (int): to reduce the size of the pointcloud
        mask ( Optional[np.ndarray]): default to all ones mask
        bounds (float): bounds of the point cloud to clip to in meters

    Returns:
        Tuple[np.ndarray, np.ndarray]: points, colors
    """

    HEIGHT, WIDTH = depth.shape

    assert subsample >= 1 and isinstance(
        subsample, int
    ), "subsample must be a positive int"

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    points = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

    if mask is None:
        # default to full mask
        mask = np.ones((HEIGHT, WIDTH), dtype=bool)

    U, V = np.ix_(
        np.arange(HEIGHT), np.arange(WIDTH)
    )  # pylint: disable=unbalanced-tuple-unpacking
    Z = depth.copy()

    X = (V - cx) * Z / fx
    Y = (U - cy) * Z / fy

    points[:, :, 0] = X
    points[:, :, 1] = Y
    points[:, :, 2] = Z

    colors = image.copy()

    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    # subsample
    points = points[::subsample, :]
    colors = colors[::subsample, :]

    points = points.clip(-bounds, bounds)

    return (points, colors)


def create_transform(
    x: float, y: float, z: float, roll: float, pitch: float, yaw: float
) -> np.ndarray:
    """Creates a 4x4 transformation matrix.

    This function takes the following arguments:
        x (float): The x translation in meters.
        y (float): The y translation in meters.
        z (float): The z translation in meters.
        roll (float): The roll angle in degrees.
        pitch (float): The pitch angle in degrees.
        yaw (float): The yaw angle in degrees.

    Returns:
        A 4x4 numpy array representing the transformation matrix.
    """
    # Convert degrees to radians
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    # Create individual rotation matrices
    R_yaw = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    R_pitch = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )

    R_roll = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ]
    )

    # Combine the rotation matrices
    rotation_matrix = R_yaw @ R_pitch @ R_roll

    # Construct the transformation matrix
    transformation_matrix = np.array(
        [
            [
                rotation_matrix[0, 0],
                rotation_matrix[0, 1],
                rotation_matrix[0, 2],
                x,
            ],
            [
                rotation_matrix[1, 0],
                rotation_matrix[1, 1],
                rotation_matrix[1, 2],
                y,
            ],
            [
                rotation_matrix[2, 0],
                rotation_matrix[2, 1],
                rotation_matrix[2, 2],
                z,
            ],
            [0, 0, 0, 1],
        ]
    )

    return transformation_matrix


###############################################################################


def interpolate_trajectory_3D(
    trajectory: np.ndarray,
    samples: int = 0,
) -> np.ndarray:
    """
    Interpolates the trajectory (N, 3) to (M, 3)
    Where M = N*(S+1)+1

    :param trajectory: (N,3) numpy trajectory
    :type trajectory: np.ndarray
    :param samples: number of samples
    :type samples: int
    :returns: (M,3) interpolated numpy trajectory
    """
    if trajectory.shape[0] == 0:
        return trajectory
    # Calculate the number of segments
    num_segments = trajectory.shape[0] - 1

    # Generate the interpolated trajectory
    interpolated_trajectory = np.zeros((num_segments * (samples + 1) + 1, 3))

    # Fill in the interpolated points
    for i in range(num_segments):
        start = trajectory[i]
        end = trajectory[i + 1]
        interpolated_trajectory[
            i * (samples + 1) : (i + 1) * (samples + 1)
        ] = np.linspace(start, end, samples + 2)[:-1]

    # Add the last point
    interpolated_trajectory[-1] = trajectory[-1]

    return interpolated_trajectory


def plot_points_on_image(
    frame_img: np.ndarray,
    points_2D: np.ndarray,  # Shape (N, 2)
    color: Tuple[int, int, int] = (0, 255, 0),
    draw_line: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """
    Plot 2D points onto an image, optionally connecting them with lines.

    Parameters:
        frame_img: Image to plot on
        points_2D: 2D points in image space, shape (N, 2)
        color: RGB color tuple for the points and lines
        draw_line: Whether to connect points with lines
        thickness: Thickness of points and lines

    Returns:
        Image with plotted points
    """
    # Create output image (don't modify input)
    result_img = frame_img.copy()
    h, w = result_img.shape[:2]

    # Draw points
    for i, (px, py) in enumerate(points_2D):
        # Skip invalid points
        if not (np.isfinite(px) and np.isfinite(py)):
            continue

        # Skip points outside image
        if not (0 <= px < w and 0 <= py < h):
            continue

        # Draw point
        point = (int(px), int(py))
        cv2.circle(result_img, point, thickness, color, -1)

        # Draw line to previous point if requested
        if draw_line and i > 0:
            prev_px, prev_py = points_2D[i - 1]

            # Skip if previous point is invalid or outside image
            if not (np.isfinite(prev_px) and np.isfinite(prev_py)):
                continue

            prev_point = (int(prev_px), int(prev_py))
            cv2.line(result_img, prev_point, point, color, thickness)

    return result_img


def plot_3D_trajectory(
    frame_img: np.ndarray,
    trajectory_3D: np.ndarray,  # Shape (N, 3)
    intrinsic_matrix: np.ndarray,  # Shape (3, 3)
    extrinsic_matrix: np.ndarray,  # Shape (4, 4)
    color: Tuple[int, int, int] = (0, 255, 0),
    draw_line: bool = False,
    thickness: int = 2,
    interpolation_samples: int = 0,
) -> np.ndarray:
    """
    Plot a 3D trajectory onto an image.

    Coordinate frames:
    - 3D world coordinates:
        x: horizontal right
        y: vertical down
        z: depth into the camera (forward)
    - Image coordinates:
        u: horizontal right (pixels)
        v: vertical down (pixels)

    Parameters:
        frame_img: Image to plot on
        trajectory_3D: 3D trajectory points, shape (N, 3)
        intrinsic_matrix: Camera intrinsic matrix (3x3)
        extrinsic_matrix: Camera extrinsic matrix (4x4)
        color: RGB color tuple for the trajectory
        draw_line: Whether to connect points with lines
        thickness: Thickness of points and lines

    Returns:
        Image with plotted trajectory
    """
    # Validate inputs
    if not (trajectory_3D.ndim == 2 and trajectory_3D.shape[1] == 3):
        raise ValueError(
            f"Trajectory must have shape (N, 3), got {trajectory_3D.shape}"
        )

    if intrinsic_matrix.shape != (3, 3):
        raise ValueError(
            f"Intrinsic matrix must have shape (3, 3), got {intrinsic_matrix.shape}"  # noqa
        )

    if extrinsic_matrix.shape != (4, 4):
        raise ValueError(
            f"Extrinsic matrix must have shape (4, 4), got {extrinsic_matrix.shape}"  # noqa
        )

    if draw_line:
        assert (
            interpolation_samples > 0
        ), "draw_line requires a supersampled trajectory"

    trajectory_3D = interpolate_trajectory_3D(
        trajectory_3D, samples=interpolation_samples
    )

    # Transform points to camera space
    trajectory_cam = apply_transform(trajectory_3D, extrinsic_matrix)

    # Filter points behind the camera
    in_front_mask = (
        trajectory_cam[:, 2] > 0.0
    )  # Small threshold to avoid near-zero z issues
    trajectory_cam_filtered = trajectory_cam[in_front_mask]

    if len(trajectory_cam_filtered) == 0:
        return frame_img.copy()  # Nothing to draw

    # Project 3D points to image
    trajectory_2D = project_world_cam_to_image(
        trajectory_cam_filtered, intrinsic_matrix
    )

    # Use the plot_points_on_image function to draw the points
    return plot_points_on_image(
        frame_img,
        trajectory_2D,
        color=color,
        draw_line=draw_line,
        thickness=thickness,
    )
