#!/usr/bin/env python
# coding:utf-8

import os

import torch
import yaml


def check_gpu_config(gpu_id: int) -> torch.device:
    """
    Check if the specified GPU ID is available and return the appropriate device (GPU or CPU).

    Args:
        gpu_id (int): ID of the GPU to use.

    Returns:
        torch.device: Appropriate device (GPU or CPU).
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > gpu_id:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


def make_directory(dir_name: str) -> bool:
    """
    Create a new directory if the specified directory does not exist.

    Args:
        dir_name (str): Name of the directory to create.

    Returns:
        bool: True if the directory was created, False if it already exists.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        return True
    return False


def load_yaml_config(config_path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def update_args_with_yaml(args: object, yaml_config: dict) -> object:
    """
    Update command-line arguments with values from YAML configuration.

    Args:
        args (object): Command-line arguments.
        yaml_config (dict): YAML configuration dictionary.

    Returns:
        object: Updated command-line arguments.
    """
    for key, value in yaml_config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args
