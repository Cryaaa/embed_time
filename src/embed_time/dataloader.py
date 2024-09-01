from embed_time.model import ResNet2D
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from typing import List
from itertools import product
import csv
import cv2
import glob
import h5py
import numpy as np
import os
import re
import torch
import torch.nn as nn
import tqdm


class PumpingDataset(Dataset):

    def __init__(
        self,
        dataset_names,
        chunk_size,
        cycle_size,
        device
    ):
        # note: important to record the order of dataset name sequence
        self.dataset_names = dataset_names
        self.chunk_size = chunk_size
        self.cycle_size = cycle_size
        self.device = torch.device(device)

        self.x_path = "/home/alicia/store1/alicia/pumping/unaug_data"
        self.y_path = "/home/alicia/store1/alicia/pumping/pumping_labels"

        self.data_indices = self.get_data_indices()

    def get_data_indices(self):

        # list of tuples (ds_idx, chunk_idx)
        data_indices = []
        for ds_idx, dataset_name in enumerate(self.dataset_names):

            with h5py.File(f"{self.x_path}/{dataset_name}.h5", "r") as f:
                num_frames = f['img_nir'].shape[0]

            num_chunks = num_frames // self.chunk_size
            data_indices += list(product([ds_idx], list(range(num_chunks))))

        return data_indices

    def __len__(self):
        # returns total number of chunks/samples
        return len(self.data_indices)

    def __getitem__(self, index):

        ds_idx, chunk_idx = self.data_indices[index]

        # read NIR image
        start = chunk_idx * self.chunk_size
        end = start + self.chunk_size 
        dataset_name = self.dataset_names[ds_idx]

        with h5py.File(f"{self.x_path}/{dataset_name}.h5", "r") as f:
            img = torch.tensor(
                    f['img_nir'][:],
                    dtype=torch.float32,
                    device=self.device
            )
            num_frames = img.shape[0]
            img_chunk = img[slice(start, end)] 

        with open(f"{self.y_path}/{dataset_name}.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            pump_times = [int(float(row[1]) / 50) for row in reader]

        clicks = torch.zeros(num_frames, dtype=torch.float32)
        pump_times = [time for time in pump_times if time > 0]
        clicks[pump_times] = 1

        label_embeds = self.embed(clicks, num_frames)[slice(start, end)]

        return img_chunk, label_embeds, (ds_idx, chunk_idx)

    def embed(self, clicks, num_frames):

        """
        Args:
            clicks (list[int]): time frames where click events are recorded
            chunk_size (int): number of frames contained in one sample
            cycle_size (int): distance from the current time frame for embedding
        """
        embedded_labels = torch.zeros(
                (num_frames, 2),
                dtype=torch.float32,
                device=self.device
        )
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

                embedded_labels[start + i] = torch.tensor(
                        [y1, y2],
                        dtype=torch.float32,
                        device=self.device
                )

        return embedded_labels


if __name__ == "__main__":

    device = 'cuda:2'
    cycle_size = 3
    chunk_size = 10
    batch_size = 32
    input_channels = chunk_size
    output_classes = chunk_size * 2

    dataset_names = ['2023-01-16-01', '2022-06-28-07', '2023-01-17-14', '2023-01-05-18']
    dataset = PumpingDataset(
            dataset_names=dataset_names,
            chunk_size=chunk_size,
            cycle_size=cycle_size,
            device=device
    )
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    for i, (x, labels, item_idx) in enumerate(dataloader):

        print(f"chunked x: {x.shape}")
        print(f"chunked labels: {labels.shape}")
        print(f"item index: {item_idx}")

        model = ResNet2D(
                output_classes=output_classes,
                input_channels=input_channels,
                batch_size=batch_size,
        ).to(device)
        out = model(x)
        print(out)
        print(out.shape)
        break

