#!/usr/bin/env python
# coding:utf-8

import argparse

from detector.detector import HumanDetector
from utils.utils import load_yaml_config, update_args_with_yaml


def main(args):
    # Initialize the detector
    human_detector = HumanDetector(args)

    # Start detecting
    human_detector.run()


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Detect human program")

    # Add command line arguments
    parser.add_argument("--config", type=str, default="config/detect.yml", help="Path to the YAML configuration file")
    parser.add_argument("--input_type", type=str, choices=["image", "movie"], help="Input type: 'image' or 'movie'")
    parser.add_argument("--image_list_file", type=str, help="Path to the image list file")
    parser.add_argument("--movie_path", type=str, help="Path to the movie file")
    parser.add_argument("--model_path", type=str, help="Path to the ONNX model file")

    # Parse the command line arguments
    args = parser.parse_args()

    # Load the YAML configuration file
    config_yml = load_yaml_config(args.config)

    # Update the command line arguments with the YAML configuration
    args = update_args_with_yaml(args, config_yml)

    # Call the main function with the updated arguments
    main(args)
