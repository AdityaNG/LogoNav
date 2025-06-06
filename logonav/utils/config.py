"""Configuration utilities for LogoNav."""

import os
from typing import Any, Dict, Optional

import yaml


def get_default_config_path() -> str:
    """
    Get the path to the default LogoNav configuration file

    Returns:
        str: Path to the default config file
    """
    # Config file is in the same directory as this module
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "LogoNav.yaml"
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load LogoNav configuration from a YAML file

    Args:
        config_path: Path to the config file. If None, use the default config

    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        config_path = get_default_config_path()

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config
