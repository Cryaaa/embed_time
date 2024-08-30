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

    # approach to label samples in a batch
    # 'labels': [(ds_indx, chunk_idx)]
    # e.g. [(0, 0), .., (0, 9), (1, 0), ...]
    # each label is attached to `x` and `embeddings`

    def __init__ (
        self,
        dataset_names,
        chunk_size,
        cycle_size
    ):
        self.dataset_names = dataset_names
        self.chunk_size = chunk_size
        self.cycle_size = cycle_size
        # TODO: chunk + normalize data beforehand and store in new directories
        self.x_path = "/home/alicia/store1/alicia/pumping/raw_data"
        self.y_path = "/home/alicia/store1/alicia/pumping/pumping_labels"

        self.images, self.label_embeddings, self.labels = self.process()
        assert len(self.images) == len(self.label_embeddings) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        item_idx = self.labels[index] # (ds_idx, chunk_idx)
        x = self.images[index]
        label_embeds = self.label_embeddings[index]

        return x, label_embeds, item_idx

    def process(self):

        images = []
        label_embeddings = []
        labels = []

        # return data tensor of shape (n, w, h, t)
        for ds_idx, dataset_name in enumerate(self.dataset_names):

            h5_file = f"{self.x_path}/{dataset_name}.h5"
            csv_file = f"{self.y_path}/{dataset_name}.csv"

            if os.path.exists(h5_file) and os.path.exists(csv_file):
                x, label_embeds = self.assemble_single(dataset_name, h5_file, csv_file)
                print(f"image shape: {x.shape}, label len: {len(label_embeds)}")
                chunked_frames, chunked_embeds, labels = self.chunk_single_ds(
                        x,
                        label_embeds,
                        ds_idx,
                        labels
                )
                print(f"num chunked frames: {len(chunked_frames)}")
                images += chunked_frames
                label_embeddings += chunked_embeds

        return images, label_embeddings, labels

    def chunk_single_ds(self, x, label_embeds, ds_idx, labels):

        num_frames = x.shape[0]
        chunked_frames = []
        chunked_embeds = []

        for chunk_idx, start_frame in enumerate(range(0, num_frames, self.chunk_size)):

            end_frame = start_frame + self.chunk_size

            if end_frame < num_frames - 1:
                chunk_x = x[start_frame:end_frame] / 255
                chunk_y = label_embeds[start_frame:end_frame]
                #print(f"chunked labels: {chunk_y.shape}")
                chunked_frames.append(chunk_x)
                chunked_embeds.append(chunk_y)
                labels.append((ds_idx, chunk_idx))

        return chunked_frames, chunked_embeds, labels

    def assemble_single(self, dataset_name, h5_file, csv_file):

        """ find the time frames where click events happen """

        with h5py.File(h5_file, "r") as f:
            # TODO: coarse crop
            x = torch.tensor(f["img_nir"][:, 174:558, 292:676], dtype=torch.float32)

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            pump_times = [int(float(row[1]) / 50) for row in reader]

        num_frames = x.shape[0]
        clicks = torch.zeros(num_frames, dtype=torch.float32)
        pump_times = [time for time in pump_times if time > 0]
        print(f"num clicks: {len(pump_times)}")
        clicks[pump_times] = 1
        label_embeds = self.embed(clicks, num_frames)
        print(f"label_embeds shape: {label_embeds.shape}")
        # TODO: finer crop and augment x
        return x, label_embeds

    def embed(self, clicks, num_frames):

        """
        Args:
            clicks (list[int]): time frames where click events are recorded
            chunk_size (int): number of frames contained in one sample
            cycle_size (int): distance from the current time frame for embedding
        """
        embedded_labels = torch.zeros((num_frames, 2), dtype=torch.float32)
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
        dataset_names=["2022-06-28-01", "2023-01-23-28"],
        chunk_size=120,
        cycle_size=3)

    dataloader = torch.utils.data.DataLoader(
        pumping_dataset,
        batch_size=4,
        shuffle=True,
    )

    for i, (x, labels, item_idx) in enumerate(dataloader):
        print(f"chunked x: {x.shape}")
        print(f"chunked labels: {labels.shape}")
        print(f"item index: {item_idx}")
