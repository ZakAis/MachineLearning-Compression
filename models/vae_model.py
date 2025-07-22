import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class OceanVAE(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(OceanVAE, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        # Dimensions d'entrée et de sortie
        self.input_height = 336
        self.input_width = 720
        
        # Calcul précis des dimensions latentes
        # 336×720 -> 168×360 -> 84×180 -> 42×90
        self.latent_height = 42  # CORRECTION: 42 au lieu de 43
        self.latent_width = 90
        
        # ENCODEUR
        # 336×720 → 168×360
        self.enc_conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(64)
        
        # 168×360 → 84×180  
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(128)
        
        # 84×180 → 42×90
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(256)
        
        # Projection vers l'espace latent avec initialisation appropriée
        self.to_mu = nn.Conv2d(256, latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv2d(256, latent_dim, kernel_size=1)
        
        # INITIALISATION
        nn.init.xavier_uniform_(self.to_mu.weight)
        nn.init.constant_(self.to_mu.bias, 0)
        nn.init.xavier_uniform_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, 0)  # Commencer avec variance raisonnable
        
        # DÉCODEUR
        self.from_latent = nn.Conv2d(latent_dim, 256, kernel_size=1)
        self.dec_bn0 = nn.BatchNorm2d(256)
        
        # 42×90 → 84×180 
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(128)
        
        # 84×180 → 168×360
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(64)
        
        # 168×360 → 336×720 
        self.dec_conv3 = nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1)
        
        # Initialisation des poids du décodeur
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation appropriée des poids"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def apply_mask(self, x, mask):
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            
            mask = mask.expand(x.size(0), x.size(1), -1, -1)
            x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x_masked = x_masked * mask
            return x_masked, mask
        else:
            # Gérer les NaN même sans masque
            x_clean = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            return x_clean, None
        
    def encode(self, x, mask):
        if mask is not None:
            x, full_mask = self.apply_mask(x, mask)
        else:
            x, full_mask = self.apply_mask(x, None)
        
        # Encodage avec activations plus douces
        x = F.leaky_relu(self.enc_bn1(self.enc_conv1(x)), 0.2)  # 64 × 168 × 360
        x = F.leaky_relu(self.enc_bn2(self.enc_conv2(x)), 0.2)  # 128 × 84 × 180
        x = F.leaky_relu(self.enc_bn3(self.enc_conv3(x)), 0.2)  # 256 × 42 × 90
        
        mu = self.to_mu(x)      # latent_dim × 42 × 90
        logvar = self.to_logvar(x)  # latent_dim × 42 × 90
        
        # Clamp logvar pour éviter les valeurs extrêmes
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mu, logvar, full_mask

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Mode déterministe en évaluation

    def decode(self, z):
        # Décodage avec activations cohérentes
        x = F.leaky_relu(self.dec_bn0(self.from_latent(z)), 0.2)  # 256×42×90
        x = F.leaky_relu(self.dec_bn1(self.dec_conv1(x)), 0.2)    # 128×84×180
        x = F.leaky_relu(self.dec_bn2(self.dec_conv2(x)), 0.2)    # 64×168×360
        
        # Dernière couche 
        x = self.dec_conv3(x)  # input_channels×336×720
        x = torch.tanh(x)
        
        return x

    def forward(self, x, mask=None):
        mu, logvar, processed_mask = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
        
    def get_latent_spatial_shape(self):
        return (self.latent_dim, self.latent_height, self.latent_width)
    
    def encode_to_spatial(self, x, mask):
        mu, logvar, _ = self.encode(x, mask)
        return self.reparameterize(mu, logvar)
    
    def generate(self, num_samples=1):
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, self.latent_height, self.latent_width, device=device)
        return self.decode(z)
