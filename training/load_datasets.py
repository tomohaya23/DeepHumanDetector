#!/usr/bin/env python
# coding:utf-8

import os


def load_datasets(datasets):
    """
    Load datasets from specified dataset directories.

    Args:
        datasets (list): List of dataset names.

    Returns:
        tuple: Tuple containing training data, training labels, validation data, and validation labels.
    """

    train_data = []
    train_label = []
    val_data = []
    val_label = []

    for dataset_name in datasets:
        if dataset_name == "dataset1":
            from dataset.dataset1.load_dataset1 import load_data

            train_data1, train_label1, val_data1, val_label1 = load_data(os.path.join("dataset", dataset_name))
            train_data.extend(train_data1)
            train_label.extend(train_label1)
            val_data.extend(val_data1)
            val_label.extend(val_label1)

        if dataset_name == "dataset2":
            from dataset.dataset2.load_dataset2 import load_data

            train_data2, train_label2 = load_data(os.path.join("dataset", dataset_name))
            train_data.extend(train_data2)
            train_label.extend(train_label2)

        # Add your own dataset here
        # For example:
        if dataset_name == "your_dataset":
            from dataset.your_dataset.load_dataset import load_data

            train_data_your, train_label_your = load_data(os.path.join("dataset", dataset_name))
            train_data.extend(train_data_your)
            train_label.extend(train_label_your)

    return train_data, train_label, val_data, val_label
