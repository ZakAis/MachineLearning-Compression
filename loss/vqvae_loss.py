import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import numpy as nps
import matplotlib.pyplot as plt

class OceanVQVAELoss(nn.Module):
    """
    Loss function spécialisée pour VQ-VAE océanique.
    Combine la loss de reconstruction avec la VQ loss (commitment + embedding).
    """
    def __init__(self, commitment_weight):
        """
        Args:
            commitment_weight (float): Poids de la commitment loss (équivalent du beta)
        """
        super(OceanVQVAELoss, self).__init__()
        self.commitment_weight = commitment_weight
        
    def forward(self, x_recon, x, vq_loss, ocean_mask=None):
        """
        Calcule la perte VQ-VAE combinée avec vérification stricte du format des tenseurs.
        
        Paramètres:
        -----------
        x_recon : torch.Tensor
            Données reconstruites [B, C, H, W]
        x : torch.Tensor
            Données originales [B, C, H, W]
        vq_loss : torch.Tensor
            VQ loss (commitment + embedding) calculée par le VectorQuantizer
        ocean_mask : torch.Tensor, optional
            Masque océanique binaire (1=océan, 0=continent) [H, W] ou [B, H, W] ou [B, C, H, W]
            
        Retourne:
        ---------
        total_loss : torch.Tensor
            Perte totale
        recon_loss : torch.Tensor
            Perte de reconstruction
        vq_loss_weighted : torch.Tensor
            VQ loss pondérée
        """
        # Vérification des dimensions
        if x.dim() != 4 or x_recon.dim() != 4:
            raise ValueError(f"Format incorrect: x_recon={x_recon.shape}, x={x.shape}. Attendu [B, C, H, W]")
        
        batch_size, channels, height, width = x.size()
        
        # 1. Traitement du masque océanique (identique au VAE)
        if ocean_mask is not None:
            # S'assurer que le masque a la bonne forme
            if ocean_mask.dim() == 2:  # [H, W]
                spatial_mask = ocean_mask
            elif ocean_mask.dim() == 3:  # [B, H, W] ou [C, H, W]
                spatial_mask = ocean_mask.float().any(dim=0)
            elif ocean_mask.dim() == 4:  # [B, C, H, W]
                spatial_mask = ocean_mask.float().any(dim=(0, 1))
            else:
                raise ValueError(f"Format de masque invalide: {ocean_mask.shape}")
            
            # Vérifier les dimensions
            if spatial_mask.size(0) != height or spatial_mask.size(1) != width:
                raise ValueError(f"Dimensions du masque ({spatial_mask.shape}) ne correspondent pas aux données ({height}, {width})")
            
            # Vérification que le masque contient des valeurs
            num_ocean_points = torch.sum(spatial_mask)
            if num_ocean_points == 0:
                print("ATTENTION: Masque océanique vide!")
                # Retourner une perte avec gradient
                return torch.tensor(0.0, device=x.device, requires_grad=True), \
                       torch.tensor(0.0, device=x.device, requires_grad=True), \
                       torch.tensor(0.0, device=x.device, requires_grad=True)
            
            # Créer un masque 4D [B, C, H, W]
            new_mask = spatial_mask.reshape(1, 1, height, width).expand(batch_size, channels, -1, -1)
            
            # Gérer les NaN dans les données d'entrée
            # Utiliser torch.where pour préserver les gradients
            x_clean = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x_recon_clean = torch.where(torch.isnan(x_recon), torch.zeros_like(x_recon), x_recon)
            
            # Appliquer le masque
            masked_recon = x_recon_clean * new_mask
            masked_x = x_clean * new_mask
            
            # Calculer MSE uniquement sur les zones océaniques
            ocean_points = torch.sum(new_mask)
            error = (masked_recon - masked_x)**2
            recon_loss = torch.sum(error) / ocean_points
        else:
            # Gérer les NaN même sans masque
            x_clean = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x_recon_clean = torch.where(torch.isnan(x_recon), torch.zeros_like(x_recon), x_recon)
            
            # Calcul standard de la perte MSE
            recon_loss = F.mse_loss(x_recon_clean, x_clean, reduction='mean')
        
        # 2. VQ Loss (pas de modification nécessaire, déjà calculée par le VectorQuantizer)
        # Le VQ loss contient déjà commitment_loss + embedding_loss
        # On applique juste le poids global si différent de celui du VectorQuantizer
        vq_loss_weighted = vq_loss  # Déjà pondérée dans le VectorQuantizer
        
        # 3. Perte totale
        total_loss = recon_loss + vq_loss_weighted
        
        return total_loss, recon_loss, vq_loss_weighted