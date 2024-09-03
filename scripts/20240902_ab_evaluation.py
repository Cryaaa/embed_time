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
import umap
from embed_time.model_VAE_resnet18 import VAEResNet18
from datasets.neuromast import NeuromastDatasetTest, NeuromastDatasetTrain_T10



def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch']
#%%
# Model Evaluation Function
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = total_mse = total_kld = 0
    all_latent_vectors = []
    all_metadata = []
    
    with torch.no_grad():
        for idx, (batch, label) in enumerate(dataloader):
            data = batch.to(device)
            metadata = label
            
            recon_batch, mu, logvar = model(data)
            mse = F.mse_loss(recon_batch, data, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + kld * 1e-5
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()
            
            mu_flattened = mu.view(mu.size(0), -1)
            all_latent_vectors.append(mu_flattened.cpu())
            all_metadata.extend(metadata.tolist())
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_mse = total_mse / len(dataloader.dataset)
    avg_kld = total_kld / len(dataloader.dataset)
    latent_vectors = torch.cat(all_latent_vectors, dim=0)
    
    return avg_loss, avg_mse, avg_kld, latent_vectors, all_metadata
#%%
# Visualization Functions
def plot_reconstructions(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        batch, label = next(iter(dataloader))
        data = batch.to(device)
        recon_batch, _, _ = model(data)
        
        image_idx = np.random.randint(data.shape[0])
        original = data[image_idx].cpu().numpy()
        reconstruction = recon_batch[image_idx].cpu().numpy()
        
        fig, axes = plt.subplots(1,2, figsize=(20, 10))
        
        
        axes[0].imshow(original[0], cmap='gray')
        axes[0].set_title(f'Input_image {label[image_idx]}', fontsize=30)
        axes[0].axis('off')
        axes[1].imshow(reconstruction[0], cmap='gray')
        axes[1].set_title(f'Reconstructed_image', fontsize=30)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Image shape: {original.shape}")
        print(f"Reconstruction shape: {reconstruction.shape}")
        print(f"Original image min/max values: {original.min():.4f}/{original.max():.4f}")
        print(f"Reconstructed image min/max values: {reconstruction.min():.4f}/{reconstruction.max():.4f}")
#%%
def create_pca_plots(train_latents, val_latents, train_df, val_df):
    # Step 1: Perform PCA
    pca = PCA(n_components=2)
    train_latents_pca = pca.fit_transform(train_latents)
    val_latents_pca = pca.transform(val_latents)
    
    # Step 2: Prepare the plot
    fig, axes = plt.subplots(1,2, figsize=(25, 10))
    
    # Helper function to create a color map
    def create_color_map(n):
        return ListedColormap(plt.cm.viridis(np.linspace(0, 1, n)))
    # Assuming you have 3 unique labels
    
    # Step 3: Plot PCA for the training set
    ax = axes[0]
    scatter = ax.scatter(train_latents_pca[:, 0], train_latents_pca[:, 1], c=train_df['Labels'], cmap=create_color_map(len(np.unique(train_df['Labels']))),s=100)
    ax.set_title('PCA of Training Latents', fontsize=40)
    ax.set_xlabel('PCA Component 1', fontsize=40)
    ax.set_ylabel('PCA Component 2', fontsize=40)
    # Create a color bar with specific ticks and labels
    num_labels = len(np.unique(train_df['Labels']))
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['1-SC', '2-MC', '3-HC'], fontsize=40)
    

    # Step 4: Plot PCA for the validation set
    ax = axes[1]
    scatter = ax.scatter(val_latents_pca[:, 0], val_latents_pca[:, 1], c=val_df['Labels'], cmap=create_color_map(len(np.unique(val_df['Labels']))),s=100)
    ax.set_title('PCA of Validation Latents', fontsize=40)
    ax.set_xlabel('PCA Component 1', fontsize=40)
    ax.set_ylabel('PCA Component 2', fontsize=40)
    num_labels = len(np.unique(val_df['Labels']))
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['1-SC', '2-MC', '3-HC'], fontsize=40)
    


    # Optional: You can add more plots or subplots as required

    # Debugging: Print shapes and check if the data is non-empty
    print(f"Train Latents PCA shape: {train_latents_pca.shape}")
    print(f"Val Latents PCA shape: {val_latents_pca.shape}")
    print(f"Unique labels in training set: {np.unique(train_df['Labels'])}")
    print(f"Unique labels in validation set: {np.unique(val_df['Labels'])}")
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Step 5: Show the plot
    plt.show()
#%%
def create_umap_plots(train_latents, val_latents, train_df, val_df):
    

    # Initialize UMAP
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    # Fit and transform the training data
    train_latents_umap = umap_reducer.fit_transform(train_latents)
    # Transform the validation data using the same UMAP model
    val_latents_umap = umap_reducer.transform(val_latents)

    fig, axes = plt.subplots(1,2, figsize=(25, 10))
    
    def create_color_map(n):
        return ListedColormap(plt.cm.viridis(np.linspace(0, 1, n)))

    
    # Step 5: Plot UMAP for the training set
    ax = axes[0]
    scatter = ax.scatter(train_latents_umap[:, 0], train_latents_umap[:, 1], c=train_df['Labels'], cmap=create_color_map(len(np.unique(train_df['Labels']))),s=100)
    ax.set_title('UMAP of Training Latents', fontsize=40)
    ax.set_xlabel('UMAP Component 1', fontsize=40)
    ax.set_ylabel('UMAP Component 2', fontsize=40)
    # Create a color bar with specific ticks and labels
    num_labels = len(np.unique(train_df['Labels']))
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['1-SC', '2-MC', '3-HC'], fontsize=40)
    

    # Step 6: Plot UMAP for the validation set
    ax = axes[1]
    scatter = ax.scatter(val_latents_umap[:, 0], val_latents_umap[:, 1], c=val_df['Labels'], cmap=create_color_map(len(np.unique(val_df['Labels']))),s=100)
    ax.set_title('UMAP of Validation Latents', fontsize=40)
    ax.set_xlabel('UMAP Component 1', fontsize=40)
    ax.set_ylabel('UMAP Component 2', fontsize=40)
    num_labels = len(np.unique(val_df['Labels']))
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['1-SC', '2-MC', '3-HC'], fontsize=40)
    


#%%
# Main Execution
if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model initialization and loading
    model = VAEResNet18(nc = 1, z_dim = 22 ).to(device)
    checkpoint_dir = "/mnt/efs/dlmbl/G-et/checkpoints/static/Akila/20240903z_dim-22_lr-0.0001_beta-1e-07/_epoch_6/"
    
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    model, epoch = load_checkpoint(checkpoint_path, model, device)
    model = model.to(device)

    

    
    dataset_train = NeuromastDatasetTrain_T10()
    dataset_val = NeuromastDatasetTest()
    

    dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=8)
    dataloader_val = DataLoader(dataset_val, batch_size=2, shuffle=True, num_workers=8)

    # Model evaluation
    print("Evaluating on training data...")
    train_loss, train_mse, train_kld, train_latents, train_metadata = evaluate_model(model, dataloader_train, device)
    print(f"Training - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, KLD: {train_kld:.4f}")

    print("Evaluating on validation data...")
    val_loss, val_mse, val_kld, val_latents, val_metadata = evaluate_model(model, dataloader_val, device)
    print(f"Validation - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, KLD: {val_kld:.4f}")

    # Create DataFrames
    train_df = pd.DataFrame(train_metadata, columns=['Labels'])
    train_df = pd.concat([train_df, pd.DataFrame(train_latents.numpy())], axis=1)

    val_df = pd.DataFrame(val_metadata, columns=['Labels'])
    val_df = pd.concat([val_df, pd.DataFrame(val_latents.numpy())], axis=1)
#%%
    # Visualizations
    plot_reconstructions(model, dataloader_val, device)
    plot_reconstructions(model, dataloader_train, device)
   

#%%
create_pca_plots(train_latents.numpy(), val_latents.numpy(), train_df, val_df)
#%%
create_umap_plots(train_latents.numpy(), val_latents.numpy(), train_df, val_df)
# %%
