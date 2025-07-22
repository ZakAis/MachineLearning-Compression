import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import numpy as nps
import matplotlib.pyplot as plt

class OceanWAELoss(nn.Module):
    """
    Loss function spécialisée pour WAE-MMD océanique.
    Compatible avec ton interface existante.
    """
    def __init__(self, lambda_reg, sigma_list):
        super(OceanWAELoss, self).__init__()
        self.lambda_reg = lambda_reg
        
        if sigma_list is None:
            self.sigma_list = [0.1, 1.0, 10.0]
        else:
            self.sigma_list = sigma_list
        
        print(f" WAE-MMD Loss: λ={lambda_reg}, σ={self.sigma_list}")
        
    def forward(self, x_recon, x, z, ocean_mask=None):
        """
        Interface compatible avec ton OceanVQVAELoss
        
        Args:
            x_recon: Reconstruction [B, C, H, W]
            x: Original [B, C, H, W]  
            z: Latent codes [B, D, H', W']
            ocean_mask: Masque océanique
        """
        
        # 1. RECONSTRUCTION LOSS (ton code existant)
        if ocean_mask is not None:
            # Traitement masque identique à ton VQ-VAE
            batch_size, channels, height, width = x.size()
            
            if ocean_mask.dim() == 2:
                spatial_mask = ocean_mask
            elif ocean_mask.dim() == 3:
                spatial_mask = ocean_mask.float().any(dim=0)
            elif ocean_mask.dim() == 4:
                spatial_mask = ocean_mask.float().any(dim=(0, 1))
            else:
                raise ValueError(f"Format de masque invalide: {ocean_mask.shape}")
            
            if spatial_mask.size(0) != height or spatial_mask.size(1) != width:
                raise ValueError(f"Dimensions du masque ({spatial_mask.shape}) != données ({height}, {width})")
            
            num_ocean_points = torch.sum(spatial_mask)
            if num_ocean_points == 0:
                print("ATTENTION: Masque océanique vide!")
                return torch.tensor(0.0, device=x.device, requires_grad=True), \
                       torch.tensor(0.0, device=x.device, requires_grad=True), \
                       torch.tensor(0.0, device=x.device, requires_grad=True)
            
            # Masque 4D
            new_mask = spatial_mask.reshape(1, 1, height, width).expand(batch_size, channels, -1, -1)
            
            # Nettoyage NaN
            x_clean = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x_recon_clean = torch.where(torch.isnan(x_recon), torch.zeros_like(x_recon), x_recon)
            
            # MSE masquée
            masked_recon = x_recon_clean * new_mask
            masked_x = x_clean * new_mask
            
            ocean_points = torch.sum(new_mask)
            error = (masked_recon - masked_x)**2
            recon_loss = torch.sum(error) / ocean_points
        else:
            # MSE standard
            x_clean = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x_recon_clean = torch.where(torch.isnan(x_recon), torch.zeros_like(x_recon), x_recon)
            recon_loss = F.mse_loss(x_recon_clean, x_clean, reduction='mean')
        
        # 2. MMD LOSS
        z_flat = z.view(z.size(0), -1)  # Flatten pour MMD
        mmd_loss = self.compute_mmd(z_flat)
        
        # 3. LOSS TOTALE
        total_loss = recon_loss + self.lambda_reg * mmd_loss
        
        return total_loss, recon_loss, mmd_loss
    
    def compute_mmd(self, z_samples):
        """Calcul MMD avec multiples kernels"""
        z_prior = torch.randn_like(z_samples)
        
        total_mmd = 0.0
        for sigma in self.sigma_list:
            k_zz = self.rbf_kernel(z_samples, z_samples, sigma)
            k_pp = self.rbf_kernel(z_prior, z_prior, sigma)
            k_zp = self.rbf_kernel(z_samples, z_prior, sigma)
            
            mmd = k_zz.mean() + k_pp.mean() - 2 * k_zp.mean()
            total_mmd += mmd
            
        return total_mmd / len(self.sigma_list)
    
    def rbf_kernel(self, x, y, sigma):
        """Noyau RBF pour MMD"""
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        
        x = x.unsqueeze(1)  # [B, 1, D]
        y = y.unsqueeze(0)  # [1, B, D]
        
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input / (2 * sigma**2))

