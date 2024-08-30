import os
import numpy as np
import zarr
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class ZarrCellDataset(Dataset):
    def __init__(self, parent_dir:Path, data_split:Path, channels=[0, 1, 2, 3],
                 cell_cycle_stages="interphase", transform="masks", crop_size=None):
        self.parent_dir = Path(parent_dir)
        self.gene_name = gene_name
        self.genes = list(self.parent_dir.glob("*.zarr"))
        self.barcode_name = barcode_name
        self.channels = channels
        self.cell_cycle_stages = cell_cycle_stages
        self.transform = transform
        self.crop_size = crop_size

        self.zarr_data = self._load_zarr_data()
        self.original_images, self.cell_masks, self.nuclei_masks = self._load_images_and_masks()
        self.cell_images, self.nuclei_images = self._apply_transform()

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):
        sample = {
            'gene': self.gene_name,
            'barcode': self.barcode_name,
            'stage': self.cell_cycle_stages,
            'original_image': self.original_images[idx],
            'cell_mask': self.cell_masks[idx],
            'nuclei_mask': self.nuclei_masks[idx],
            'cell_image': self.cell_images[idx],
            'nuclei_image': self.nuclei_images[idx]
        }
        return sample

    def _read_zattrs(self, path):
        zattrs = {}
        zattrs_path = os.path.join(path, ".zattrs")
        if os.path.exists(zattrs_path):
            with open(zattrs_path, "r") as f:
                zattrs = json.load(f)
        return zattrs

    def _load_zarr_data(self):
        zarr_file_gene = os.path.join(self.parent_dir, f"{self.gene_name}.zarr")
        if not os.path.isdir(zarr_file_gene):
            raise ValueError(f"Gene {zarr_file_gene} does not exist")
        
        zarr_file_barcode = os.path.join(zarr_file_gene, self.barcode_name)
        if not os.path.isdir(zarr_file_barcode):
            raise ValueError(f"Barcode {zarr_file_barcode} does not exist")

        zarr_file_stage = os.path.join(zarr_file_barcode, self.cell_cycle_stages)
        if not os.path.isdir(zarr_file_stage):
            raise ValueError(f"Stage {zarr_file_stage} does not exist")

        self._read_zattrs(zarr_file_stage)  # You might want to do something with zattrs
        
        return zarr.open(zarr_file_gene, mode='r')

    def _load_images_and_masks(self):
        original_images = self.zarr_data[self.barcode_name][self.cell_cycle_stages]['images'][:, self.channels, :, :]
        cell_masks = self.zarr_data[self.barcode_name][self.cell_cycle_stages]['cells']
        nuclei_masks = self.zarr_data[self.barcode_name][self.cell_cycle_stages]['nuclei']

        if len(original_images) != len(cell_masks) or len(original_images) != len(nuclei_masks):
            raise ValueError("Number of images, cells, and nuclei are not the same")

        cell_masks = np.expand_dims(cell_masks, 1)
        nuclei_masks = np.expand_dims(nuclei_masks, 1)

        return original_images, cell_masks, nuclei_masks

    def _apply_transform(self):
        if self.transform == "masks":
            cell_images = self.original_images * self.cell_masks
            nuclei_images = self.original_images * self.nuclei_masks
        else:
            raise ValueError("Only 'masks' is supported for transform")

        if self.crop_size is not None:
            center_height, center_width = self.original_images.shape[2] // 2, self.original_images.shape[3] // 2
            slice_h = slice(center_height - self.crop_size // 2, center_height + self.crop_size // 2)
            slice_w = slice(center_width - self.crop_size // 2, center_width + self.crop_size // 2)
            cell_images = cell_images[:, :, slice_h, slice_w]
            nuclei_images = nuclei_images[:, :, slice_h, slice_w]

        return cell_images, nuclei_images

