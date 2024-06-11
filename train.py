#!/usr/bin/env python
# coding:utf-8

import argparse

from training.trainer import HumanDetectionTrainer
from utils.utils import load_yaml_config, update_args_with_yaml


def main(args):
    # Initialize the trainer
    trainer = HumanDetectionTrainer(args)

    # Start training
    trainer.run()


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Training program for human detection model")

    # Add command line arguments
    parser.add_argument("--config", type=str, default="config/train.yml", help="Path to the YAML configuration file")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["dataset1", "dataset2", "your_dataset"],
        required=True,
        help="Select the datasets to use (multiple choices allowed)",
    )
    parser.add_argument("--total_epochs", type=int, default=30, help="Total number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--early_stopping", type=int, default=3, help="Number of epochs for early stopping")
    parser.add_argument("--input_height", type=int, default=480, help="Input image height")
    parser.add_argument("--input_width", type=int, default=640, help="Input image width")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--mosaic", type=bool, default=True, help="Apply mosaic data augmentation")
    parser.add_argument("--mixup", type=bool, default=True, help="Apply mixup data augmentation")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--latest_checkpoint", type=str, default="", help="Latest checkpoint file")
    parser.add_argument(
        "--start_epoch", type=int, default=1, help="Starting epoch (set to the epoch number to resume training from)"
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Load the YAML configuration file
    config_yml = load_yaml_config(args.config)

    # Update the command line arguments with the YAML configuration
    args = update_args_with_yaml(args, config_yml)

    # Call the main function with the updated arguments
    main(args)
