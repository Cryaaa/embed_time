#%%
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import v2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import yaml

from embed_time.dataset_static import ZarrCellDataset
from embed_time.dataloader_static import collate_wrapper
from embed_time.model_VAE_resnet18_linear import VAEResNet18_Linear

# Utility Functions
def read_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    mean = [float(i) for i in config['Dataset mean'][0].split()]
    std = [float(i) for i in config['Dataset std'][0].split()]
    return np.array(mean), np.array(std)

def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch']

# Model Evaluation Function
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = total_mse = total_kld = 0
    all_latent_vectors = []
    all_metadata = []
    
    with torch.no_grad():
        for batch in dataloader:
            data = batch['cell_image'].to(device)
            metadata = [batch['gene'], batch['barcode'], batch['stage']]
            
            recon_batch, _, mu, logvar = model(data)
            mse = F.mse_loss(recon_batch, data, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + kld * 1e-5
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()
            
            all_latent_vectors.append(mu.cpu())
            all_metadata.extend(zip(*metadata))
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_mse = total_mse / len(dataloader.dataset)
    avg_kld = total_kld / len(dataloader.dataset)
    latent_vectors = torch.cat(all_latent_vectors, dim=0)
    
    return avg_loss, avg_mse, avg_kld, latent_vectors, all_metadata

# Visualization Functions
def plot_reconstructions(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        data = batch['cell_image'].to(device)
        recon_batch, _, _, _ = model(data)
        
        image_idx = np.random.randint(data.shape[0])
        original = data[image_idx].cpu().numpy()
        reconstruction = recon_batch[image_idx].cpu().numpy()
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        for j in range(4):
            axes[0, j].imshow(original[j], cmap='gray')
            axes[0, j].set_title(f'Original Channel {j+1}')
            axes[0, j].axis('off')
            axes[1, j].imshow(reconstruction[j], cmap='gray')
            axes[1, j].set_title(f'Reconstructed Channel {j+1}')
            axes[1, j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Image shape: {original.shape}")
        print(f"Reconstruction shape: {reconstruction.shape}")
        print(f"Original image min/max values: {original.min():.4f}/{original.max():.4f}")
        print(f"Reconstructed image min/max values: {reconstruction.min():.4f}/{reconstruction.max():.4f}")

def create_pca_plots(train_latents, val_latents, train_df, val_df):
    pca = PCA(n_components=2)
    train_latents_pca = pca.fit_transform(train_latents)
    val_latents_pca = pca.transform(val_latents)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    def create_color_map(n):
        return ListedColormap(plt.cm.tab20(np.linspace(0, 1, n)))

    attributes = ['stage', 'barcode', 'gene']
    for i, attr in enumerate(attributes):
        for j, (latents_pca, df) in enumerate([(train_latents_pca, train_df), (val_latents_pca, val_df)]):
            unique_values = df[attr].unique()
            color_map = create_color_map(len(unique_values))
            color_dict = {value: i for i, value in enumerate(unique_values)}
            colors = [color_dict[value] for value in df[attr]]
            
            scatter = axes[j, i].scatter(latents_pca[:, 0], latents_pca[:, 1], c=colors, s=5, cmap=color_map)
            axes[j, i].set_title(f"{'Training' if j == 0 else 'Validation'} Latent Space (PCA) - Colored by {attr}")
            axes[j, i].set_xlabel("PC1")
            axes[j, i].set_ylabel("PC2")
            
            cbar = plt.colorbar(scatter, ax=axes[j, i], ticks=range(len(unique_values)))
            cbar.set_ticklabels(unique_values)

    plt.tight_layout()
    plt.show()
#%%
# Main Execution
if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model initialization and loading
    model = VAEResNet18_Linear(nc=4, z_dim=72, input_spatial_dim=[96,96])
    checkpoint_dir = "/mnt/efs/dlmbl/G-et/checkpoints/static/Matteo/20240902_1450_resnet_linear_test/"
    checkpoints = sorted(os.listdir(checkpoint_dir), key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
    model, epoch = load_checkpoint(checkpoint_path, model, device)
    model = model.to(device)
    print(model)

    # Dataset parameters
    parent_dir = '/mnt/efs/dlmbl/S-md/'
    csv_file = '/mnt/efs/dlmbl/G-et/csv/dataset_split_2.csv'
    channels = [0, 1, 2, 3]
    transform = "masks"
    crop_size = 96
    normalizations = v2.Compose([v2.CenterCrop(crop_size)])
    yaml_file_path = "/mnt/efs/dlmbl/G-et/yaml/dataset_info_20240901_155625.yaml"
    dataset_mean, dataset_std = read_config(yaml_file_path)

    # Dataset and DataLoader creation
    metadata_keys = ['gene', 'barcode', 'stage']
    images_keys = ['cell_image']
    
    dataset_train = ZarrCellDataset(parent_dir, csv_file, 'train', channels, transform, normalizations, None, dataset_mean, dataset_std)
    dataset_val = ZarrCellDataset(parent_dir, csv_file, 'val', channels, transform, normalizations, None, dataset_mean, dataset_std)

    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=8, collate_fn=collate_wrapper(metadata_keys, images_keys))
    dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=True, num_workers=8, collate_fn=collate_wrapper(metadata_keys, images_keys))

    # Model evaluation
    print("Evaluating on training data...")
    train_loss, train_mse, train_kld, train_latents, train_metadata = evaluate_model(model, dataloader_train, device)
    print(f"Training - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, KLD: {train_kld:.4f}")

    print("Evaluating on validation data...")
    val_loss, val_mse, val_kld, val_latents, val_metadata = evaluate_model(model, dataloader_val, device)
    print(f"Validation - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, KLD: {val_kld:.4f}")

    # Create DataFrames
    train_df = pd.DataFrame(train_metadata, columns=['gene', 'barcode', 'stage'])
    train_df = pd.concat([train_df, pd.DataFrame(train_latents.numpy())], axis=1)

    val_df = pd.DataFrame(val_metadata, columns=['gene', 'barcode', 'stage'])
    val_df = pd.concat([val_df, pd.DataFrame(val_latents.numpy())], axis=1)

    # Visualizations
    plot_reconstructions(model, dataloader_val, device)
    plot_reconstructions(model, dataloader_train, device)
    create_pca_plots(train_latents.numpy(), val_latents.numpy(), train_df, val_df)

#%%