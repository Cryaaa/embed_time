from embed_time.model import ResNet2D
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
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
        cycle_size,
        device
    ):
        self.dataset_names = dataset_names
        self.chunk_size = chunk_size
        self.cycle_size = cycle_size
        self.device = torch.device(device)

        self.x_path = "/home/alicia/store1/alicia/pumping/unaug_data"
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

                chunked_frames, chunked_embeds, labels = self.chunk_single_ds(
                        x,
                        label_embeds,
                        ds_idx,
                        labels
                )

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
                chunk_x = x[start_frame:end_frame]
                chunk_y = label_embeds[start_frame:end_frame]

                chunked_frames.append(chunk_x)
                chunked_embeds.append(chunk_y)
                labels.append((ds_idx, chunk_idx))

        return chunked_frames, chunked_embeds, labels

    def assemble_single(
            self,
            dataset_name,
            h5_file,
            csv_file,
    ):

        """ find the time frames where click events happen """

        with h5py.File(h5_file, "r") as f:
            x = torch.tensor(
                    f["img_nir"][:],
                    dtype=torch.float32,
                    device=self.device
            )

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            pump_times = [int(float(row[1]) / 50) for row in reader]

        num_frames = x.shape[0]
        clicks = torch.zeros(num_frames, dtype=torch.float32)
        pump_times = [time for time in pump_times if time > 0]
        clicks[pump_times] = 1
        label_embeds = self.embed(clicks, num_frames)

        return x, label_embeds

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

    pumping_dataset = PumpingDataset(
        dataset_names=["2022-06-28-01"],
        chunk_size=chunk_size,
        cycle_size=cycle_size,
        device=device
    )

    dataloader = torch.utils.data.DataLoader(
        pumping_dataset,
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

