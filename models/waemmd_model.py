import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np

# ==========================================
# MODÈLE WAE-MMD OCÉANIQUE
# ==========================================

class OceanWAE(nn.Module):
    """
    Wasserstein Auto-Encoder avec Maximum Mean Discrepancy (WAE-MMD)
    pour les données océaniques avec masquage spatial.
    Basé sur l'architecture d'OceanFullMapVAE mais adapté pour WAE-MMD.
    """
    
    def __init__(self,
                 input_channels: int,  # T, S, U, V
                 latent_dim: int,      # 16 canaux latents
                 hidden_dims: List[int],
                 lambda_reg: float,
                 sigma_list: List[float],
                 **kwargs) -> None:
        super(OceanWAE, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lambda_reg = lambda_reg
        self.sigma_list = sigma_list
        
        # Dimensions fixes basées sur votre architecture
        self.latent_height = 42    # dimension spatiale hauteur
        self.latent_width = 90     # dimension spatiale largeur
        self.input_height = 336
        self.input_width = 720
        
        # ENCODEUR (identique au VAE mais sans mu/logvar)
        # 336×720 → 168×360
        self.enc_conv1 = nn.Conv2d(input_channels, hidden_dims[0], kernel_size=4, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(hidden_dims[0])
        
        # 168×360 → 84×180  
        self.enc_conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=4, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(hidden_dims[1])
        
        # 84×180 → 42×90
        self.enc_conv3 = nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=4, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(hidden_dims[2])
        
        # Projection directe vers l'espace latent (pas de mu/logvar comme dans VAE)
        self.to_latent = nn.Conv2d(hidden_dims[2], latent_dim, kernel_size=1)
        
        # DÉCODEUR (symétrique exact au VAE)
        self.from_latent = nn.Conv2d(latent_dim, hidden_dims[2], kernel_size=1)
        self.dec_bn0 = nn.BatchNorm2d(hidden_dims[2])
        
        # 42×90 → 84×180 
        self.dec_conv1 = nn.ConvTranspose2d(hidden_dims[2], hidden_dims[1], kernel_size=4, stride=2, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(hidden_dims[1])
        
        # 84×180 → 168×360
        self.dec_conv2 = nn.ConvTranspose2d(hidden_dims[1], hidden_dims[0], kernel_size=4, stride=2, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(hidden_dims[0])
        
        # 168×360 → 336×720 
        self.dec_conv3 = nn.ConvTranspose2d(hidden_dims[0], input_channels, kernel_size=4, stride=2, padding=1)
    
    def _ensure_same_device(self, *tensors):
        """
        Utilitaire pour s'assurer que tous les tenseurs sont sur le même device
        """
        if not tensors:
            return tensors
        
        # Prendre le device du premier tenseur non-None
        reference_device = None
        for tensor in tensors:
            if tensor is not None:
                reference_device = tensor.device
                break
        
        if reference_device is None:
            return tensors
        
        # Déplacer tous les tenseurs vers le device de référence
        result = []
        for tensor in tensors:
            if tensor is not None:
                result.append(tensor.to(reference_device))
            else:
                result.append(None)
        
        return result
    
    def apply_mask(self, x, mask):
        """
        Applique le masque océanique aux données d'entrée
        """
        # S'assurer que x et mask sont sur le même device
        x, mask = self._ensure_same_device(x, mask)
        
        # Nettoyer TOUS les NaN dès l'entrée
        x_clean = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        mask = mask.expand(x_clean.size(0), x_clean.size(1), -1, -1)
        x_masked = x_clean * mask
        
        return x_masked, mask
    
    def encode(self, x, mask=None):
        """
        Encode l'input vers l'espace latent (comme le VAE)
        """
        # S'assurer que x et mask sont sur le même device
        x, mask = self._ensure_same_device(x, mask)
        
        if mask is not None:
            x, full_mask = self.apply_mask(x, mask)
        else:
            # Nettoyage même sans masque (protection supplémentaire)
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        # Encodage avec downsampling précis
        x = F.leaky_relu(self.enc_bn1(self.enc_conv1(x)), 0.2)
        x = F.leaky_relu(self.enc_bn2(self.enc_conv2(x)), 0.2)
        x = F.leaky_relu(self.enc_bn3(self.enc_conv3(x)), 0.2)
        
        # Projection directe vers l'espace latent
        z = self.to_latent(x)
        
        return z
    
    def decode(self, z):
        """
        Décode depuis l'espace latent vers l'espace des données
        """
        # Décodage symétrique avec upsampling précis
        x = F.leaky_relu(self.dec_bn0(self.from_latent(z)), 0.2)
        x = F.leaky_relu(self.dec_bn1(self.dec_conv1(x)), 0.2)
        x = F.leaky_relu(self.dec_bn2(self.dec_conv2(x)), 0.2)
        x = torch.tanh(self.dec_conv3(x))
        
        return x
    
    def reconstruction_loss(self, recon, target, mask=None):
        """
        Calcule la loss de reconstruction (MSE) avec masquage optionnel
        """
        # Nettoyage silencieux des NaN
        if torch.isnan(recon).any():
            recon = torch.where(torch.isnan(recon) | torch.isinf(recon), torch.zeros_like(recon), recon)
        
        if torch.isnan(target).any():
            target = torch.where(torch.isnan(target) | torch.isinf(target), torch.zeros_like(target), target)
        
        if mask is not None:
            # S'assurer que le masque est sur le bon device
            _, mask = self._ensure_same_device(recon, mask)
            
            # Expand le masque si nécessaire
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
                mask = mask.expand(recon.size(0), recon.size(1), -1, -1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
                mask = mask.expand(-1, recon.size(1), -1, -1)
            
            # Calculer la MSE seulement sur les zones océaniques
            diff = (recon - target) ** 2
            masked_diff = diff * mask.float()
            
            # Éviter division par zéro
            mask_sum = mask.float().sum()
            if mask_sum < 1:
                return torch.tensor(1e-3, device=recon.device, requires_grad=True)
            
            loss = masked_diff.sum() / mask_sum
        else:
            # MSE standard
            loss = F.mse_loss(recon, target)
        
        # Vérification finale
        if torch.isnan(loss):
            return torch.tensor(1e-3, device=recon.device, requires_grad=True)
        
        return loss
    
    def mmd_loss(self, x, y, kernel='multiscale'):
        """
        Calcule la Maximum Mean Discrepancy entre deux distributions
        """
        # S'assurer que x et y sont sur le même device
        x, y = self._ensure_same_device(x, y)
        
        # Stabilisation 1: Vérifier les inputs
        if torch.isnan(x).any() or torch.isnan(y).any():
            print(" NaN détecté dans les inputs MMD")
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        
        def rbf_kernel_multiscale_stable(x1, x2, sigma_list):
            """Noyau RBF multi-échelle STABILISÉ"""
            # Calculer les distances euclidiennes au carré avec stabilisation
            x1_norm = (x1**2).sum(dim=1, keepdim=True)
            x2_norm = (x2**2).sum(dim=1, keepdim=True)
            
            dist_sq = x1_norm + x2_norm.t() - 2 * torch.mm(x1, x2.t())
            
            # Stabilisation 2: Clamp les distances pour éviter les valeurs négatives
            dist_sq = torch.clamp(dist_sq, min=0.0)
            
            # Sommer sur tous les sigmas avec stabilisation
            K = torch.zeros_like(dist_sq)
            for sigma in sigma_list:
                # Stabilisation 3: Éviter sigma trop petit
                sigma_safe = max(sigma, 1e-6)
                exp_term = -dist_sq / (2 * sigma_safe**2)
                
                # Stabilisation 4: Clamp l'exponentielle
                exp_term = torch.clamp(exp_term, min=-50, max=50)  # Éviter overflow/underflow
                K += torch.exp(exp_term)
            
            return K / len(sigma_list)
        
        # Tailles des échantillons
        n_x = x.size(0)
        n_y = y.size(0)
        
        if kernel == 'multiscale':
            # Calcul des termes de la MMD avec noyau multi-échelle
            K_xx = rbf_kernel_multiscale_stable(x, x, self.sigma_list)
            K_xy = rbf_kernel_multiscale_stable(x, y, self.sigma_list)
            K_yy = rbf_kernel_multiscale_stable(y, y, self.sigma_list)
            
            # Stabilisation 5: Vérifier les noyaux
            if torch.isnan(K_xx).any() or torch.isnan(K_xy).any() or torch.isnan(K_yy).any():
                print(" NaN détecté dans les noyaux")
                return torch.tensor(0.0, device=x.device, requires_grad=True)
            
            # MMD² = E[K(x,x)] - 2E[K(x,y)] + E[K(y,y)]
            mmd_sq = (K_xx.sum() / (n_x * n_x) - 
                     2 * K_xy.sum() / (n_x * n_y) + 
                     K_yy.sum() / (n_y * n_y))
            
            # Stabilisation 6: Clamp MMD² avant racine carrée
            mmd_sq = torch.clamp(mmd_sq, min=1e-12)
            
        else:
            raise ValueError(f"Noyau {kernel} non supporté")
        
        # Stabilisation 7: Racine carrée sécurisée
        mmd_result = torch.sqrt(mmd_sq)
        
        # Staibilisation 8: Vérification finale
        if torch.isnan(mmd_result):
            print(" NaN dans le résultat MMD final")
            return torch.tensor(1e-6, device=x.device, requires_grad=True)
        
        return mmd_result
    
    def forward(self, input_tensor, mask=None):
        """
        Forward pass complet du WAE-MMD
        """
        # S'assurer que input_tensor et mask sont sur le même device
        input_tensor, mask = self._ensure_same_device(input_tensor, mask)
        
        # Sauvegarder une version propre du target
        clean_target = torch.where(torch.isnan(input_tensor) | torch.isinf(input_tensor), 
                                 torch.zeros_like(input_tensor), input_tensor)
        
        # Encoder
        z = self.encode(input_tensor, mask)
        
        # Decoder
        recon = self.decode(z)
        
        # Application du masque à la reconstruction si fourni
        if mask is not None:
            recon, _ = self.apply_mask(recon, mask)
        
        # Retour avec target nettoyé
        return recon, clean_target, z
    
    def loss_function(self, recon, target, z, mask=None):
        """
        Calcule la loss totale du WAE-MMD
        
        Returns: Dictionnaire compatible avec train_wae
        """
        # Calcul de la loss de reconstruction
        recon_loss = self.reconstruction_loss(recon, target, mask)
        
        # Calcul de la MMD loss
        z_flat = z.view(z.size(0), -1)
        prior_samples = torch.randn_like(z_flat)
        mmd_loss = self.mmd_loss(z_flat, prior_samples)
        
        # Loss totale avec lambda_reg du constructeur
        total_loss = recon_loss + self.lambda_reg * mmd_loss
        
        # Retour en dictionnaire
        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'mmd_loss': mmd_loss
        }
    
    def get_latent_spatial_shape(self):
        """
        Retourne la forme spatiale de l'espace latent
        (Identique au VAE)
        """
        return (self.latent_dim, self.latent_height, self.latent_width)
    
    def encode_to_spatial(self, x, mask=None):
        """
        Encode vers l'espace latent spatial
        (Simplifié par rapport au VAE car pas de reparameterization)
        """
        return self.encode(x, mask)
    
    def generate(self, num_samples=1):
        """
        Génère de nouveaux échantillons en échantillonnant depuis la distribution prior
        (Identique au VAE)
        """
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, self.latent_height, self.latent_width, device=device)
        return self.decode(z)
    
    def sample(self, num_samples, device='cpu'):
        """
        Alias pour generate (compatibilité)
        """
        return self.generate(num_samples).to(device)
    
    def interpolate(self, x1, x2, mask1=None, mask2=None, num_steps=10):
        """
        Interpole entre deux échantillons dans l'espace latent
        """
        self.eval()
        with torch.no_grad():
            # S'assurer que les inputs sont sur le même device
            x1, mask1 = self._ensure_same_device(x1, mask1)
            x2, mask2 = self._ensure_same_device(x2, mask2)
            
            # Encoder les deux échantillons
            z1 = self.encode(x1, mask1)
            z2 = self.encode(x2, mask2)
            
            # Créer l'interpolation linéaire
            interpolations = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                # Décoder
                recon_interp = self.decode(z_interp)
                interpolations.append(recon_interp)
            
        return torch.stack(interpolations)
    
    def get_model_info(self):
        """
        Retourne les informations sur le modèle
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'WAE-MMD',
            'input_channels': self.input_channels,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'lambda_reg': self.lambda_reg,
            'sigma_list': self.sigma_list,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': f"({self.input_channels}, {self.input_height}, {self.input_width})",
            'latent_shape': f"({self.latent_dim}, {self.latent_height}, {self.latent_width})"
        }