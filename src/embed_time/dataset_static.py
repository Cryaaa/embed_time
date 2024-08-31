import os
import numpy as np
import zarr
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd

class ZarrCellDataset(Dataset):
    def __init__(self, parent_dir, csv_file, split = "train", channels=[0, 1, 2, 3], transform="masks", crop_size=None):
        self.parent_dir = Path(parent_dir)
        self.channels = channels
        self.transform = transform
        self.crop_size = crop_size

        # Load the CSV file
        self.data_info = pd.read_csv(csv_file)

        # Split the data into train, val, and test
        if split == "train":
            self.data_info = self.data_info[self.data_info['split'] == "train"]
        elif split == "val":
            self.data_info = self.data_info[self.data_info['split'] == "val"]
        elif split == "test":
            self.data_info = self.data_info[self.data_info['split'] == "test"]

        # Group the data by gene, barcode, and stage
        self.grouped_data = self.data_info.groupby(['gene', 'barcode', 'stage'])

        # Load all zarr data
        self.zarr_data = self._load_all_zarr_data()

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        gene = row['gene']
        barcode = row['barcode']
        stage = row['stage']
        cell_idx = row['cell_idx']

        # Get the zarr data for this gene, barcode, and stage
        zarr_group = self.zarr_data[(gene, barcode, stage)]

        # Load images and masks
        original_image = zarr_group['images'][cell_idx]
        original_image = original_image[self.channels]  # Select specified channels
        cell_mask = zarr_group['cells'][cell_idx]
        nuclei_mask = zarr_group['nuclei'][cell_idx]

        # Apply transform
        cell_image, nuclei_image = self._apply_transform(original_image, cell_mask, nuclei_mask)

        sample = {
            'gene': gene,
            'barcode': barcode,
            'stage': stage,
            'cell_idx': cell_idx,
            'split': row['split'],
            'original_image': original_image,
            'cell_mask': cell_mask,
            'nuclei_mask': nuclei_mask,
            'cell_image': cell_image,
            'nuclei_image': nuclei_image
        }

        return sample

    def _load_all_zarr_data(self):
        zarr_data = {}
        for (gene, barcode, stage), group in self.grouped_data:
            zarr_file = self.parent_dir / f"{gene}.zarr" / barcode / stage
            if not zarr_file.is_dir():
                raise ValueError(f"Zarr file not found: {zarr_file}")
            zarr_data[(gene, barcode, stage)] = zarr.open(zarr_file, mode='r')
        return zarr_data

    def _apply_transform(self, original_image, cell_mask, nuclei_mask):

        if self.transform == "masks":
            cell_image = original_image * cell_mask
            nuclei_image = original_image * nuclei_mask
        else:
            raise ValueError("Only 'masks' is supported for transform")

        if self.crop_size is not None:
            center_height, center_width = original_image.shape[1] // 2, original_image.shape[2] // 2
            slice_h = slice(center_height - self.crop_size // 2, center_height + self.crop_size // 2)
            slice_w = slice(center_width - self.crop_size // 2, center_width + self.crop_size // 2)
            cell_image = cell_image[:, slice_h, slice_w]
            nuclei_image = nuclei_image[:, slice_h, slice_w]

        return cell_image, nuclei_image

class ZarrCellDataset_specific(Dataset):
    def __init__(self, parent_dir, gene_name, barcode_name, channels=[0, 1, 2, 3],
                 cell_cycle_stages="interphase", transform="masks", crop_size=None):
        self.parent_dir = parent_dir
        self.gene_name = gene_name
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

