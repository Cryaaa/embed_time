from torch.utils.data import DataLoader, Dataset
from typing import List
import csv
import cv2
import glob
import h5py
import numpy as np
import os
import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm


class PumpingDataset(Dataset):

    def __init__ (
        self,
        dataset_names,
        chunk_size,
        cycle_size
    ):
        self.dataset_names = dataset_names
        self.chunk_size = chunk_size
        self.cycle_size = cycle_size
        self.x_path = "/home/alicia/store1/alicia/pumping/raw_data"
        self.y_path = "/home/alicia/store1/alicia/pumping/pumping_labels"
        self.datasets = self.process()

    def __getitem__(self, index):
        x, y = self.datasets[index][index]
        return x, y

    def __len__(self):
        return len(self.datasets)

    def process(self):
        # return data tensor of shape (n, w, h, t)
        datasets = []
        for dataset_name in self.dataset_names:

            h5_file = f"{self.x_path}/{dataset_name}.h5"
            csv_file = f"{self.y_path}/{dataset_name}.csv"

            if os.path.exists(h5_file) and os.path.exists(csv_file):
                x, labels = self.assemble_single(dataset_name, h5_file, csv_file)
                print(f"image shape: {x.shape}, label len: {len(labels)}")
                chunked_frames = self.chunk_single_ds(x, labels)
                print(f"num chunked frames: {len(chunked_frames)}")
                datasets.append(chunked_frames)

        return datasets

    def chunk_single_ds(self, x, labels):

        num_frames = x.shape[0]
        chunked_frames = []

        for i in range(0, num_frames, self.chunk_size):
            chunk_x = x[i:i+self.chunk_size] / 255
            chunk_labels = labels[i:i+self.chunk_size]
            chunked_frames.append((chunk_x, chunk_labels))

        return chunked_frames

    def assemble_single(self, dataset_name, h5_file, csv_file):

        """ find the time frames where click events happen """

        with h5py.File(h5_file, "r") as f:
            x = torch.tensor(f["img_nir"][:, 174:558, 292:676], dtype=torch.float32)

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            pump_times = [int(float(row[1]) / 50) for row in reader]

        clicks = torch.zeros(x.shape[0], dtype=torch.float32)
        # non-zero pump times corresponding to frames where a click event happens
        pump_times = [time for time in pump_times if time > 0]
        clicks[pump_times] = 1

        labels = self.embed(clicks)
        # TODO: crop and augment x
        return x, labels

    def embed(self, clicks):

        """
        Args:
            clicks (list[int]): time frames where click events are recorded
            chunk_size (int): number of frames contained in one sample
            cycle_size (int): distance from the current time frame for embedding
        """
        embedded_labels = torch.zeros((self.chunk_size, 2), dtype=torch.float32)
        embedded_labels[:, 0] = 0
        embedded_labels[:, 1] = -1

        interval = 2 * self.cycle_size + 1

        for click in clicks:

            start = max(0, click - self.cycle_size)
            angles = np.linspace(0, np.pi, interval)

            for i, angle in enumerate(angles):

                if start + i == click:
                    y1 = 0
                    y2 = 1
                else:
                    y1 = np.cos(angle)
                    y2 = np.sin(angle)

                embedded_labels[start + i] = torch.tensor([y1, y2], dtype=torch.float32)

        return embedded_labels


class CentralCrop(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        _, _, width, height = x.shape
        start_x = width // 2 - 128
        start_y = height // 2 - 128

        return 2 * x[:, :, start_x:start_x+256, start_y:start_y+256] - 1


if __name__ == "__main__":

    pumping_dataset = PumpingDataset(
        dataset_names=["2022-01-17-01", "2022-06-28-01", "2023-01-23-28"],
        chunk_size=120,
        cycle_size=3)
    dataloader = torch.utils.data.DataLoader(
        pumping_dataset,
        batch_size=2,
        shuffle=True,
    )

    for idx, (x, labels) in enumerate(dataloader):
        print(f"chunked x: {x.shape}")
        print(f"chunked labels: {labels.shape}")

