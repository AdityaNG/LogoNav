"""Download utilities for LogoNav model weights."""

import shutil
import zipfile
from pathlib import Path
from typing import Optional

import gdown


def download_and_extract_model_weights(
    cache_dir_str: Optional[str] = None,
) -> str:
    """
    Download and extract model weights from Google Drive

    Args:
        cache_dir_str: Directory to store the model weights. If None, use
            ~/.cache/logonav

    Returns:
        str: Path to the extracted logonav.pth file
    """
    if cache_dir_str is None:
        cache_dir = Path.home() / ".cache" / "logonav"
    else:
        cache_dir = Path(cache_dir_str)

    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if model weights already exist
    model_path = cache_dir / "logonav.pth"
    if model_path.exists():
        print(f"Model weights found at {model_path}")
        return str(model_path)

    # Download the zip file
    zip_path = cache_dir / "model_weights.zip"
    if not zip_path.exists():
        print("Downloading model weights...")
        # Google Drive file ID for model_weights.zip
        file_id = "1zZpGoJYPhQDN_riUsJBR4O9lF4uhM5_B"
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            str(zip_path),
            quiet=False,
        )

    # Extract the zip file
    print(f"Extracting model weights to {cache_dir}...")
    with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
        zip_ref.extractall(str(cache_dir))

    # Move the file if it's in a subdirectory
    extracted_path = cache_dir / "model_weights" / "logonav.pth"
    if extracted_path.exists():
        shutil.move(str(extracted_path), str(model_path))
        # Clean up the extracted directory
        shutil.rmtree(str(cache_dir / "model_weights"))

    # Verify the file exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Failed to extract logonav.pth to {model_path}"
        )

    print(f"Model weights successfully extracted to {model_path}")
    return str(model_path)
