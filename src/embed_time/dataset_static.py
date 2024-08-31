import os
import numpy as np
import zarr
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd

class ZarrCellDataset(Dataset):
    def __init__(self, parent_dir, csv_file, split="train", channels=[0, 1, 2, 3], mask="masks", normalizations=None, interpolations=None):
        self.parent_dir = Path(parent_dir)
        self.channels = channels
        self.mask = mask
        self.normalizations = normalizations
        self.interpolations = interpolations

        self.data_info = pd.read_csv(csv_file)
        self.data_info = self.data_info[self.data_info['split'] == split]
        self.grouped_data = self.data_info.groupby(['gene', 'barcode', 'stage'])
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

        mean_channel = np.mean(original_image, axis=(-2, -1))
        std_channel = np.std(original_image, axis=(1, 2))

        # Apply mask and normalization
        cell_image, nuclei_image = self._apply_mask_normalization(original_image, cell_mask, nuclei_mask)

        # Apply interpolations
        cell_image, nuclei_image = self._apply_interpolation(cell_image, nuclei_image)

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
    
      
    def _apply_mask_normalization(self, original_image, cell_mask, nuclei_mask):
        if self.mask == "masks":
            cell_image = original_image * cell_mask
            nuclei_image = original_image * nuclei_mask
        else:
            raise ValueError("Only 'masks' is supported for mask")
        cell_image = torch.from_numpy(cell_image).float()
        nuclei_image = torch.from_numpy(nuclei_image).float()

        if self.normalizations:
            if isinstance(self.normalizations, list):
                for normalization in self.normalizations:
                    cell_image = normalization(cell_image)
                    nuclei_image = normalization(nuclei_image)
            else:
                cell_image = self.normalizations(cell_image)
                nuclei_image = self.normalizations(nuclei_image)

        return cell_image, nuclei_image

    def _apply_interpolation(self, cell_image, nuclei_image):
        if self.interpolations:
            if isinstance(self.interpolations, list):
                for interpolation in self.interpolations:
                    cell_image, nuclei_image = interpolation(cell_image, nuclei_image)
            else:
                cell_image, nuclei_image = self.interpolations(cell_image, nuclei_image)
        return cell_image, nuclei_image

class ZarrCellDataset_specific(Dataset):
    def __init__(self, parent_dir, gene_name, barcode_name, channels=[0, 1, 2, 3], cell_cycle_stages="interphase", 
                 mask="masks", normalizations=None, interpolations=None):
        self.parent_dir = parent_dir
        self.gene_name = gene_name
        self.barcode_name = barcode_name
        self.channels = channels
        self.cell_cycle_stages = cell_cycle_stages
        self.mask = mask
        self.normalizations = normalizations
        self.interpolations = interpolations

        self.zarr_data = self._load_zarr_data()
        self.original_images, self.cell_masks, self.nuclei_masks = self._load_images_and_masks()
        self.mean_channel, self.std_channel = self._compute_mean_std()

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):
        original_image = self.original_images[idx]
        cell_mask = self.cell_masks[idx]
        nuclei_mask = self.nuclei_masks[idx]

        normalized_mean = np.mean(self.mean_channel, axis=0)
        normalized_std = np.mean(self.std_channel, axis=0)

        print(normalized_mean, normalized_std)

        cell_image, nuclei_image = self._apply_mask_normalization(original_image, cell_mask, nuclei_mask, 
                                                                  normalized_mean, normalized_std)
        cell_image, nuclei_image = self._apply_interpolation(cell_image, nuclei_image)

        sample = {
            'gene': self.gene_name,
            'barcode': self.barcode_name,
            'stage': self.cell_cycle_stages,
            'original_image': original_image,
            'cell_mask': cell_mask,
            'nuclei_mask': nuclei_mask,
            'cell_image': cell_image,
            'nuclei_image': nuclei_image
        }
        return sample

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
    
    def _compute_mean_std(self):
        mean_channel = np.mean(self.original_images, axis=(-2, -1))
        std_channel = np.std(self.original_images, axis=(-2, -1))
        return mean_channel, std_channel

    def _apply_mask_normalization(self, original_image, cell_mask, nuclei_mask, mean, std):
        if self.mask == "masks":
            cell_image = original_image * cell_mask
            nuclei_image = original_image * nuclei_mask
        else:
            raise ValueError("Only 'masks' is supported for mask")
        cell_image = torch.from_numpy(cell_image).float()
        nuclei_image = torch.from_numpy(nuclei_image).float()

        if self.normalizations:
            if isinstance(self.normalizations, list):
                print("it is a list")
                for normalization in self.normalizations:
                    cell_image = normalization(cell_image)
                    nuclei_image = normalization(nuclei_image)
            else:
                cell_image = self.normalizations(cell_image)
                nuclei_image = self.normalizations(nuclei_image)

        return cell_image, nuclei_image

    def _apply_interpolation(self, cell_image, nuclei_image):
        if self.interpolations:
            if isinstance(self.interpolations, list):
                for interpolation in self.interpolations:
                    cell_image, nuclei_image = interpolation(cell_image, nuclei_image)
            else:
                cell_image, nuclei_image = self.interpolations(cell_image, nuclei_image)
        return cell_image, nuclei_image

    def _read_zattrs(self, path):
        zattrs = {}
        zattrs_path = os.path.join(path, ".zattrs")
        if os.path.exists(zattrs_path):
            with open(zattrs_path, "r") as f:
                zattrs = json.load(f)
        return zattrs