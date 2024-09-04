#%%
from embed_time.evaluate_static import ModelEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch 
import re

import re

def parse_checkpoint_dir(checkpoint_dir):
    filename = checkpoint_dir.split('/')[-1]
    print(filename)
    params = ['model', 'crop_size', 'nc', 'z_dim', 'lr', 'beta', 'transform', 'loss']
    result = {}
    model_match = re.search(r'_(VAE_ResNet18)_', filename)
    if model_match:
        result['model'] = model_match.group(1)
    
    # Extract other parameters
    for param in params:
        if param == 'model':
            continue  # we've already handled this
        match = re.search(rf'{param}_([^_]+)', filename)
        if match:
            value = match.group(1)
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            result[param] = value
    
    if 'benchmark' in filename:
        result['csv_file'] = 'dataset_split_benchmark.csv'
    
    return result

# model checkpoint directory
checkpoint_dir = '/mnt/efs/dlmbl/G-et/checkpoints/static/Matteo/20240903_2130_VAE_ResNet18_crop_size_64_nc_4_z_dim_30_lr_0.0001_beta_1e-05_transform_min_loss_L1_benchmark'
# variant parameters
config = parse_checkpoint_dir(checkpoint_dir)

# invariant parameters
config['checkpoint_dir'] = checkpoint_dir
config['parent_dir'] = '/mnt/efs/dlmbl/S-md/'
config['channels'] = [0, 1, 2, 3]
config['yaml_file_path'] = '/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml'
config['output_dir'] = os.path.join('/home/S-md/embed_time/scripts/latent', checkpoint_dir.split('/')[-1])
config['sampling_number'] = 3
config['csv_file'] = '/mnt/efs/dlmbl/G-et/csv/' + config['csv_file']
config['batch_size'] = 16
config['num_workers'] = 8
config['metadata_keys'] = ['gene', 'barcode', 'stage']
config['images_keys'] = ['cell_image']

# Initialize ModelEvaluator
evaluator = ModelEvaluator(config)