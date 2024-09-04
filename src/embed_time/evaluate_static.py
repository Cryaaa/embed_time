import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import v2
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import argparse
import piq

loss_ssim = piq.SSIMLoss()

from embed_time.dataset_static import ZarrCellDataset
from embed_time.dataloader_static import collate_wrapper
from embed_time.model_VAE_resnet18_linear import VAEResNet18_Linear
from embed_time.model_VAE_resnet18 import VAEResNet18
from embed_time.model import VAE, Encoder, Decoder

class ModelEvaluator():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._init_model()
        self.dataset_mean, self.dataset_std = self._read_config()
        self.output_dir = self._create_output_dir()
        self.train_df = self._evaluate('train')
        self.val_df = self._evaluate('val')

    def _init_model(self):
        model = None  # Initialize model to None
        if self.config['model'] == 'VAE_ResNet18':
            model = VAEResNet18(nc=self.config['nc'], z_dim=self.config['z_dim'])
        elif self.config['model'] == 'VAE_ResNet18_Linear':
            model = VAEResNet18_Linear(nc=self.config['nc'], z_dim=self.config['z_dim'], input_spatial_dim=self.config['input_spatial_dim'])
        elif self.config['model'] == 'VAE':
            encoder = Encoder(self.config['nc'], self.config['z_dim'])
            decoder = Decoder(self.config['z_dim'], self.config['h_dim1'], self.config['h_dim2'], self.config['nc'], self.config['output_shape'])
            model = VAE(encoder, decoder)
        else:
            raise ValueError(f"Model {self.config['model']} not supported.")
        checkpoints = sorted(os.listdir(self.config['checkpoint_dir']), key=lambda x: os.path.getmtime(os.path.join(self.config['checkpoint_dir'], x)))
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], checkpoints[-1])
        model, _ = self._load_checkpoint(checkpoint_path, model)
        return model.to(self.device)

    def _read_config(self):
        with open(self.config['yaml_file_path'], 'r') as file:
            yaml_config = yaml.safe_load(file)
        mean = [float(i) for i in yaml_config['Dataset mean'][0].split()]
        std = [float(i) for i in yaml_config['Dataset std'][0].split()]
        return np.array(mean), np.array(std)

    def _load_checkpoint(self, checkpoint_path, model):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f"Loading checkpoint from epoch {checkpoint['epoch']}...")
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint['epoch']

    def _create_dataloader(self, split):
        dataset = ZarrCellDataset(
            self.config['parent_dir'], 
            self.config['csv_file'], 
            split, 
            self.config['channels'], 
            self.config['transform'], 
            v2.Compose([v2.CenterCrop(self.config['crop_size'])]), 
            None, 
            self.dataset_mean, 
            self.dataset_std
        )
        return DataLoader(
            dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=self.config['num_workers'], 
            collate_fn=collate_wrapper(self.config['metadata_keys'], self.config['images_keys'])
        )
    
    def _create_output_dir(self):
        output_dir = os.path.join(self.config['output_dir'], 'reconstructed_images')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _evaluate_model(self, dataloader):
        self.model.eval()
        total_loss = total_mse = total_kld = 0
        all_latent_vectors = []
        all_metadata = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                data = batch['cell_image'].to(self.device)
                metadata = [batch[key] for key in self.config['metadata_keys']]
                
                if self.config['model'] == 'VAE_ResNet18_Linear':
                    recon_batch, _, mu, logvar = self.model(data)
                elif self.config['model'] == 'VAE_ResNet18':
                    recon_batch, mu, logvar = self.model(data)

                if self.config['loss'] == "MSE":
                    RECON = F.mse_loss(recon_batch, data, reduction='mean')
                elif self.config['loss'] == "L1":
                    RECON = F.l1_loss(recon_batch, data, reduction='mean')
                elif self.config['loss'] == "SSIM":
                    # normalize x for ssim (remember shape is BxCxHxW)
                    x_norm = (data - data.min()) / (data.max() - data.min())
                    recon_x_norm = (recon_batch - recon_batch.min()) / (recon_x.max() - recon_x.min())
                    ssim = loss_ssim(recon_x_norm, x_norm)
                    RECON = F.l1_loss(recon_batch, data, reduction='mean') + ssim * 0.5
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = RECON + KLD * self.config['beta']
                
                total_loss += loss.item()
                total_mse += RECON.item()
                total_kld += RECON.item()
                
                if batch_idx == 0:
                    self._save_image(data, recon_batch, self.config['output_dir'])

                if self.config['sampling_number'] > 1:
                    print('Sampling {} times...'.format(self.config['sampling_number']))
                    for i in range(self.config['sampling_number']):
                        # Sample from the latent space
                        z = self.model.reparameterize(mu, logvar)
                        # save zs and metadata into additional latent representations
                        all_latent_vectors.append(z.cpu())
                        all_metadata.extend(zip(*metadata))
                else:
                    all_latent_vectors.append(mu.cpu())
                    all_metadata.extend(zip(*metadata))
        
        avg_loss = total_loss / len(dataloader.dataset)
        avg_mse = total_mse / len(dataloader.dataset)
        avg_kld = total_kld / len(dataloader.dataset)
        latent_vectors = torch.cat(all_latent_vectors, dim=0)
        
        return avg_loss, avg_mse, avg_kld, latent_vectors, all_metadata

    def _evaluate(self, split):
        dataloader = self._create_dataloader(split)
        print(f"Evaluating on {split} data...")
        loss, mse, kld, latents, metadata = self._evaluate_model(dataloader)
        print(f"{split.capitalize()} - Loss: {loss:.4f}, MSE: {mse:.4f}, KLD: {kld:.4f}")
        
        if self.config['model'] == 'VAE_ResNet18_Linear':
            print(f"Reconstruction shape: {latents.shape}")
        elif self.config['model'] == 'VAE_ResNet18':
            # flatten the latent vectors
            latents = latents.view(latents.shape[0], -1)
            print(f"Latent shape: {latents.shape}")
        # Create DataFrame
        df = pd.DataFrame(metadata, columns=self.config['metadata_keys'])
        latent_df = pd.DataFrame(latents.numpy(), columns=[f'latent_{i}' for i in range(latents.shape[1])])
        df = pd.concat([df, latent_df], axis=1)
        # Save the latent vectors
        df.to_csv(os.path.join(self.config['output_dir'], f"{split}_{self.config['sampling_number']}_latent_vectors.csv"), index=False)
                               
        return df
    
    def _save_image(self, data, recon, output_dir):
        image_idx = np.random.randint(data.shape[0])
        original = data[image_idx].cpu().numpy()
        reconstruction = recon[image_idx].cpu().numpy()
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        channel_names = ['dapi', 'gh2ax', 'tubulin', 'actin']  # Adjust these names as needed
        
        for i in range(4):
            # Original image
            im = axes[0, i].imshow(original[i], cmap='viridis')
            axes[0, i].set_title(f'Original {channel_names[i]}', fontsize=12)
            axes[0, i].axis('off')
            fig.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # Reconstructed image
            im = axes[1, i].imshow(reconstruction[i], cmap='viridis')
            axes[1, i].set_title(f'Reconstructed {channel_names[i]}', fontsize=12)
            axes[1, i].axis('off')
            fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

        plt.tight_layout()
        
        # Create filename
        filename = f"{self.config['model']}_sample_image.png"
        
        # save the image
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory

    # 

def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
