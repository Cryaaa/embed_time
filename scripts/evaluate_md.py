from embed_time.evaluate_static import ModelEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_cell_data(original, reconstruction):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for j in range(4):
        axes[0, j].imshow(original[j], cmap='gray', vmin=-1, vmax=1)
        axes[0, j].set_title(f'Original Channel {j+1}')
        axes[0, j].axis('off')
        axes[1, j].imshow(reconstruction[j], cmap='gray', vmin=-1, vmax=1)
        axes[1, j].set_title(f'Reconstructed Channel {j+1}')
        axes[1, j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Image shape: {original.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Original image min/max values: {original.min():.4f}/{original.max():.4f}")
    print(f"Reconstructed image min/max values: {reconstruction.min():.4f}/{reconstruction.max():.4f}")

# Your configuration
config = {
    'model': 'VAEResNet18_Linear',
    'nc': 4,
    'z_dim': 32,
    'input_spatial_dim': [96, 96],
    'checkpoint_dir': "/mnt/efs/dlmbl/G-et/da_testing/training_logs/",
    'parent_dir': '/mnt/efs/dlmbl/S-md/',
    'csv_file': '/mnt/efs/dlmbl/G-et/csv/dataset_split_17_sampled.csv',
    'channels': [0, 1, 2, 3],
    'transform': "masks",
    'crop_size': 96,
    'yaml_file_path': "/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml",
    'batch_size': 16,
    'num_workers': 8,
    'metadata_keys': ['gene', 'barcode', 'stage'],
    'images_keys': ['cell_image'],
    'kld_weight': 1e-5,
    'output_dir': '/mnt/efs/dlmbl/G-et/latent_space_data/',
    'sampling_number': 3
}

# Initialize ModelEvaluator
evaluator = ModelEvaluator(config)
    
train_df = evaluator.evaluate('train')
val_df = evaluator.evaluate('val')

# save train_df and val_df to csv in graphs subdirectory
model_name = "17_genes_resnet18_linear_latent32"

os.makedirs("latent", exist_ok=True)
train_df.to_csv(f"latent/{model_name}_train.csv", index=False)
val_df.to_csv(f"latent/{model_name}_val.csv", index=False)
print(val_df.shape)

print("Evaluation complete. Latent dimensions extracted and saved.")