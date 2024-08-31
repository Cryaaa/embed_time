import os
import numpy as np
import zarr
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
from joblib import Parallel, delayed
import argparse


class DatasetSplitter:
    def __init__(self, parent_dir, output_file, train_ratio=0.7, val_ratio=0.15, num_workers=-1):
        self.parent_dir = Path(parent_dir)
        self.output_file = Path(output_file)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers

    def generate_cells_from_gene(self, gene_path):
        gene_path = Path(gene_path)
        gene_name = gene_path.stem
        cell_data = []
        
        for barcode in gene_path.glob("*"):
            barcode_name = barcode.stem
            stages = [stage for stage in barcode.glob("*") if stage.name != '.zgroup']
            
            for stage in stages:
                stage_name = stage.stem
                cells_zarr = zarr.open(stage / "images")
                
                num_cells = cells_zarr.shape[0]
                
                # Use torch to create a random permutation
                indices = torch.randperm(num_cells)
                
                # Calculate split sizes
                train_size = int(num_cells * self.train_ratio)
                val_size = int(num_cells * self.val_ratio)
                
                # Split indices
                train_indices = indices[:train_size]
                val_indices = indices[train_size:train_size+val_size]
                test_indices = indices[train_size+val_size:]
                
                # Create cell data
                for split, split_indices in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
                    for cell_idx in split_indices.tolist():
                        cell_data.append([gene_name, barcode_name, stage_name, cell_idx, split])
        
        return cell_data

    def generate_split(self):
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        genes = list(self.parent_dir.glob("*.zarr"))
        genes = ["/mnt/efs/dlmbl/S-md/AAAS.zarr", "/mnt/efs/dlmbl/S-md/AAGAB.zarr"]  # Uncomment this line to process only specific genes
        
        print(f"Processing {len(genes)} genes...")

        # Use joblib.Parallel for parallelization
        results = Parallel(n_jobs=self.num_workers, verbose=1)(
            delayed(self.generate_cells_from_gene)(gene) for gene in genes
        )
        
        print("Combining results...")
        # Flatten the list of lists
        all_cell_data = [item for sublist in results for item in sublist]

        print("Creating DataFrame and saving CSV...")
        df = pd.DataFrame(all_cell_data, columns=["gene", "barcode", "stage", "cell_idx", "split"])
        df.to_csv(self.output_file, index=False)
        print(f"Dataset split CSV saved to {self.output_file}")
