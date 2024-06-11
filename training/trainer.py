#!/usr/bin/env python
# coding:utf-8

import csv
import os
import shutil
import time

import torch
import torch.optim as optim
from models.large_model import HumanDetectionModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.dataset import HumanDetectionDataset
from training.load_datasets import load_datasets
from training.losses import HumanDetectionLoss
from utils.utils import check_gpu_config, make_directory

# from models.small_model import HumanDetectionModel


class HumanDetectionTrainer:
    def __init__(self, args):
        self.device = check_gpu_config(args.gpu_id)
        self.datasets = args.datasets
        self.total_epochs = args.total_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.early_stopping = args.early_stopping
        self.input_height = args.input_height
        self.input_width = args.input_width
        self.num_classes = args.num_classes
        self.mosaic = args.mosaic
        self.mosaic_prob = args.mosaic_prob
        self.mixup = args.mixup
        self.mixup_prob = args.mixup_prob
        self.num_workers = args.num_workers
        self.start_epoch = args.start_epoch
        self.latest_checkpoint = args.latest_checkpoint
        self.save_iteration_mode = args.save_iteration_mode
        self.columns = shutil.get_terminal_size().columns

    @staticmethod
    def custom_collate_fn(batch):
        batch_image_data, batch_label = zip(*batch)
        batch_image_data = torch.stack(batch_image_data)
        return batch_image_data, batch_label

    def create_dataset(self, data, labels, training):
        return HumanDetectionDataset(
            data,
            labels,
            self.input_width,
            self.input_height,
            self.mosaic,
            self.mosaic_prob,
            self.mixup,
            self.mixup_prob,
            training=training,
        )

    def create_data_loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn,
        )

    def train_one_epoch(self, model, optimizer, loss_func, train_loader, epoch):
        model.train()
        total_train_loss_epoch = 0
        with tqdm(train_loader, leave=False) as pbar_train_iteration:
            for iteration, batch_data in enumerate(pbar_train_iteration, 1):
                optimizer.zero_grad()
                inputs, targets = batch_data
                inputs = inputs.to(self.device)
                targets = [target.to(self.device) for target in targets]
                outputs = model(inputs)
                loss_value = loss_func(outputs, targets)
                loss_value.backward()
                optimizer.step()
                total_train_loss_epoch += loss_value.item()
                pbar_train_iteration.set_description(f"[Training epoch {epoch} iteration {iteration}]")
                pbar_train_iteration.set_postfix(loss=loss_value.item())

                # Save model every 2000 iterations
                if self.save_iteration_mode and iteration % 2000 == 0:
                    self.save_checkpoint(model, optimizer, epoch, iteration)

                time.sleep(0.1)
        total_train_loss_epoch /= iteration
        return total_train_loss_epoch

    def validate(self, model, loss_func, val_loader):
        model.eval()
        total_val_loss_epoch = 0
        with torch.no_grad():
            with tqdm(val_loader, leave=False) as pbar_val_iteration:
                for iteration, batch_data in enumerate(pbar_val_iteration, 1):
                    pbar_val_iteration.set_description("[Validating]")
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = [target.to(self.device) for target in targets]
                    outputs = model(inputs)
                    loss_value = loss_func(outputs, targets)
                    total_val_loss_epoch += loss_value.item()
                    pbar_val_iteration.set_postfix(loss=loss_value.item())
                    time.sleep(0.1)
        total_val_loss_epoch /= iteration
        return total_val_loss_epoch

    def save_checkpoint(self, model, optimizer, epoch, iteration=None):
        make_directory("exp/train/weights")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if iteration:
            torch.save(checkpoint, f"exp/train/weights/epoch{epoch}_iter{iteration}_model.pth")
        else:
            torch.save(checkpoint, f"exp/train/weights/epoch{epoch}_model.pth")

    def save_best_model(self, model, optimizer):
        make_directory("exp/train/best_model")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, "exp/train/best_model/best_model.pth")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = HumanDetectionModel(self.num_classes).to(self.device)
        model.load_state_dict(checkpoint["model"])
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.937, 0.999))
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer

    def run(self):
        train_data, train_label, val_data, val_label = load_datasets(self.datasets)
        train_dataset = self.create_dataset(train_data, train_label, training=True)
        val_dataset = self.create_dataset(val_data, val_label, training=False)

        train_loader = self.create_data_loader(train_dataset, shuffle=True)
        val_loader = self.create_data_loader(val_dataset, shuffle=False)

        if os.path.isfile(self.latest_checkpoint):
            model, optimizer = self.load_checkpoint(self.latest_checkpoint)
        else:
            model = HumanDetectionModel(self.num_classes).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.937, 0.999))

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        loss_func = HumanDetectionLoss(self.num_classes)

        train_num = len(train_dataset)
        val_num = len(val_dataset)

        best_train_loss = float("inf")
        best_val_loss = float("inf")
        no_improve = 0

        make_directory("exp/train/loss")
        loss_log_file = open("exp/train/loss/log.csv", "a", newline="")
        loss_log_writer = csv.writer(loss_log_file)
        if self.start_epoch == 1:
            loss_log_writer.writerow(["epoch", "train_loss", "valid_loss"])

        print(model)

        print("=" * self.columns)
        print(
            "[Resume Training or Fine-tuning]".center(self.columns)
            if os.path.isfile(self.latest_checkpoint)
            else "[Start Training]".center(self.columns)
        )
        print("=" * self.columns)
        print(f"Total epoch: {self.total_epochs:,}".center(self.columns))
        print(f"Learning Rate: {self.learning_rate}".center(self.columns))
        print(f"Batch size: {self.batch_size:,}".center(self.columns))
        print(f"Early stopping num: {self.early_stopping}".center(self.columns))
        print(f"Input height: {self.input_height:,}".center(self.columns))
        print(f"Input width: {self.input_width:,}".center(self.columns))
        print(f"Class num: {self.num_classes}".center(self.columns))
        print(f"Train data num: {train_num:,}".center(self.columns))
        print(f"Val data num: {val_num:,}".center(self.columns))
        print(f"Model parameter: {total_params:,}".center(self.columns))
        print("=" * self.columns)

        for epoch in range(self.start_epoch, self.total_epochs + 1):
            total_train_loss_epoch = self.train_one_epoch(model, optimizer, loss_func, train_loader, epoch)

            total_val_loss_epoch = self.validate(model, loss_func, val_loader) if val_num != 0 else 0

            print(
                f"train epoch : {epoch}, ",
                f"train_loss: {total_train_loss_epoch:.4f}, ",
                f"val_loss: {total_val_loss_epoch:.4f}",
            )

            loss_log_writer.writerow(
                [
                    f"{epoch}",
                    f"{total_train_loss_epoch}",
                    f"{total_val_loss_epoch}",
                ]
            )
            loss_log_file.flush()

            self.save_checkpoint(model, optimizer, epoch)

            if val_num != 0:
                if total_val_loss_epoch < best_val_loss:
                    best_val_loss = total_val_loss_epoch
                    self.save_best_model(model, optimizer)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if total_train_loss_epoch < best_train_loss:
                    best_train_loss = total_train_loss_epoch
                    self.save_best_model(model, optimizer)
                    no_improve = 0
                else:
                    no_improve += 1

            if self.early_stopping and no_improve >= self.early_stopping:
                print("[Early Stopping]".center(self.columns))
                break

        loss_log_file.close()
        print("[Finish Training]".center(self.columns))
