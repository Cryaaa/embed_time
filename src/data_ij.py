from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

download_path = Path(__file__).parent / "downloads"

class LiveGastruloid(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, download=False):
        """
        Parameters
        ----------
        root: Union[str, pathlib.Path]
            Data root for download; defaults to ./downloads
        train: bool
            Passed to `torchvision.datasets.MNIST`; default is True
        download: bool
            Passed to `torchvision.datasets.MNIST`; default is True
        """
        super().__init__(root, train=train, download=download)

    def __getitem__(self, item):
        image = super().__getitem__(item)
        image = transforms.ToTensor()(image)
        return image