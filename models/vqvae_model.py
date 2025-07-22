import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Union
import numpy as np

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer adapté pour les cartes océaniques.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: torch.Tensor) -> tuple:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Calculer la distance L2 entre les latents et les poids d'embeddings 
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Sélectionner l'encoding pour lequel la distance est minimale
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convertir en one-hot encoding
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantifier les latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Calculer les pertes VQ
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Remettre le résidu aux latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss, encoding_inds.view(latents_shape[0], latents_shape[1], latents_shape[2])


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.resblock(input)

# =====================================
# MODÈLE VQ-VAE
# =====================================

class OceanVQVAE(nn.Module):
    """VQ-VAE robuste avec gestion des NaN"""
    
    def __init__(self,
                 input_channels: int,
                 embedding_dim: int,
                 num_embeddings: int ,
                 hidden_dims: List,
                 beta: float,
                 **kwargs) -> None:
        super(OceanVQVAE, self).__init__()

        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        
        self.input_height = 336
        self.input_width = 720
        
        if hidden_dims is None:
            hidden_dims = [8, 16, 32] 
        
        self.latent_height = 42
        self.latent_width = 90
        
        print(f"🔧 VQ-VAE Robust: {hidden_dims} → {self.latent_height}×{self.latent_width}")

        # ============= ENCODEUR  =============
        encoder_modules = []
        in_channels = input_channels
        
        for i, h_dim in enumerate(hidden_dims):
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=False),  # inplace=False pour éviter les bugs
                    nn.Dropout2d(0.1)  # Légère régularisation
                )
            )
            in_channels = h_dim
        
        # Couches de stabilisation
        encoder_modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(0.2, inplace=False)
            )
        )

        # Couches résiduelles simplifiées
        for _ in range(1):  # Réduit pour éviter l'instabilité
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(0.2, inplace=False)
                )
            )
        
        # Projection finale
        encoder_modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1),
                nn.BatchNorm2d(embedding_dim),
                nn.Tanh()  # Limitation des valeurs
            )
        )

        self.encoder = nn.Sequential(*encoder_modules)

        # ============= VECTOR QUANTIZER =============
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, beta)

        # ============= DÉCODEUR  ===============
        decoder_modules = []
        
        decoder_modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(0.2, inplace=False)
            )
        )

        # Couches résiduelles simplifiées
        for _ in range(1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dims[-1]),
                    nn.LeakyReLU(0.2, inplace=False)
                )
            )

        # Upsampling
        hidden_dims_reversed = hidden_dims[::-1]
        
        for i in range(len(hidden_dims_reversed) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims_reversed[i], 
                                     hidden_dims_reversed[i + 1],
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims_reversed[i + 1]),
                    nn.LeakyReLU(0.2, inplace=False),
                    nn.Dropout2d(0.1)
                )
            )

        # Couche finale
        decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims_reversed[-1], 
                                 input_channels,
                                 kernel_size=4, stride=2, padding=1),
                nn.Tanh()  # Valeurs dans [-1, 1]
            )
        )

        self.decoder = nn.Sequential(*decoder_modules)
        
        # Initialisation des poids
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialisation conservative des poids"""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(module.weight, gain=0.8)  # Gain réduit
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def apply_mask(self, x, mask):
        """Version ultra-robuste"""
        if mask is None:
            # Gérer les NaN même sans masque
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x = torch.clamp(x, -10, 10)
            return x, None
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
    
        mask = mask.expand(x.size(0), x.size(1), -1, -1)
        
        # Nettoyage agressif
        x_clean = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        x_masked = x_clean * mask.float()
        
        # Normalisation douce
        x_masked = torch.clamp(x_masked, -3, 3)
        
        return x_masked, mask

    def forward(self, input: torch.Tensor, mask=None, **kwargs):
        """Forward avec vérifications à chaque étape"""
        
        # Nettoyage initial
        if mask is not None:
            input, _ = self.apply_mask(input, mask)
        else:
            input = torch.where(torch.isnan(input) | torch.isinf(input), 
                              torch.zeros_like(input), input)
        
        # Encodage
        encoding = self.encoder(input)
        
        # Vérification post-encodage
        if torch.isnan(encoding).any() or torch.isinf(encoding).any():
            print(" NaN/Inf dans l'encodage, remplacement par zéros")
            encoding = torch.where(torch.isnan(encoding) | torch.isinf(encoding),
                                 torch.zeros_like(encoding), encoding)
        
        # Quantification
        quantized_inputs, vq_loss, encoding_indices = self.vq_layer(encoding)
        
        # Vérification VQ
        if torch.isnan(vq_loss) or torch.isinf(vq_loss):
            print(" VQ Loss problématique, utilisation d'une valeur par défaut")
            vq_loss = torch.tensor(0.0, device=input.device)
        
        # Décodage
        reconstruction = self.decoder(quantized_inputs)
        
        # Vérification finale
        if torch.isnan(reconstruction).any() or torch.isinf(reconstruction).any():
            print(" NaN/Inf dans la reconstruction")
            reconstruction = torch.where(torch.isnan(reconstruction) | torch.isinf(reconstruction),
                                        torch.zeros_like(reconstruction), reconstruction)
        
        return [reconstruction, input, vq_loss, encoding_indices]

    def loss_function(self, *args, **kwargs):
        recons, input_tensor, vq_loss = args[0], args[1], args[2]
        
        # Vérifications préliminaires
        if torch.isnan(recons).any() or torch.isnan(input_tensor).any():
            return {
                'loss': torch.tensor(1000.0, device=recons.device),  # Pénalité élevée
                'Reconstruction_Loss': torch.tensor(1000.0, device=recons.device),
                'VQ_Loss': vq_loss if not torch.isnan(vq_loss) else torch.tensor(0.0, device=recons.device)
            }
        
        # Loss de reconstruction avec protection
        try:
            if 'mask' in kwargs and kwargs['mask'] is not None:
                mask = kwargs['mask']
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                elif mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                mask = mask.expand_as(input_tensor)
                
                # MSE seulement sur les zones océaniques
                recons_masked = recons * mask
                input_masked = input_tensor * mask
                
                # Protection division par zéro
                valid_pixels = mask.sum()
                if valid_pixels > 0:
                    recons_loss = F.mse_loss(recons_masked, input_masked, reduction='sum') / valid_pixels
                else:
                    recons_loss = torch.tensor(0.0, device=recons.device)
            else:
                recons_loss = F.mse_loss(recons, input_tensor)
            
            # Vérification finale
            if torch.isnan(recons_loss) or torch.isinf(recons_loss):
                recons_loss = torch.tensor(1000.0, device=recons.device)
            
            total_loss = recons_loss + vq_loss
            
            return {
                'loss': total_loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss': vq_loss
            }
            
        except Exception as e:
            print(f" Erreur dans loss_function: {e}")
            return {
                'loss': torch.tensor(1000.0, device=recons.device),
                'Reconstruction_Loss': torch.tensor(1000.0, device=recons.device),
                'VQ_Loss': vq_loss
            }

    # Autres méthodes identiques...
    def encode(self, input, mask=None):
        if mask is not None:
            input, _ = self.apply_mask(input, mask)
        return [self.encoder(input)]
    
    def decode(self, z):
        return self.decoder(z)
    
    def get_latent_shape(self):
        return (self.embedding_dim, self.latent_height, self.latent_width)


# ============= FONCTIONS UTILITAIRES =============
def debug_vqvae_forward(model, train_maps, device):
    """
    Diagnostic complet pour identifier la source des NaN en cas d'erreur y étant du
    """
    print("\n === DIAGNOSTIC COMPLET VQ-VAE ===")
    
    model.eval()
    with torch.no_grad():
        # Prendre le premier échantillon
        data, mask = train_maps[0]
        print(f" Échantillon original: {data.shape}, mask: {mask.shape}")
        
        # Statistiques des données d'entrée
        print(f" Data - Min: {data.min().item():.4f}, Max: {data.max().item():.4f}")
        print(f" Data - NaN count: {torch.isnan(data).sum().item()}")
        print(f" Mask - Ocean points: {mask.sum().item()}, Land points: {(~mask.bool()).sum().item()}")
        
        # Préparation identique à l'entraînement
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        data = data.to(device)
        mask = mask.to(device)
        
        print(f"📊 Après reshape: data={data.shape}, mask={mask.shape}")
        
        # ÉTAPE 1: Test apply_mask
        print("\n ÉTAPE 1: Test apply_mask")
        try:
            masked_data, processed_mask = model.apply_mask(data, mask)
            print(f" apply_mask OK: {masked_data.shape}")
            print(f" Masked data - Min: {masked_data.min().item():.4f}, Max: {masked_data.max().item():.4f}")
            print(f" Masked data - NaN count: {torch.isnan(masked_data).sum().item()}")
            print(f" Masked data - Inf count: {torch.isinf(masked_data).sum().item()}")
        except Exception as e:
            print(f" apply_mask FAILED: {e}")
            return False
        
        # ÉTAPE 2: Test encodeur étape par étape
        print("\n ÉTAPE 2: Test encodeur étape par étape")
        x = masked_data
        
        for i, layer in enumerate(model.encoder):
            try:
                x_before = x.clone()
                x = layer(x)
                
                has_nan = torch.isnan(x).any()
                has_inf = torch.isinf(x).any()
                
                print(f"Layer {i}: {x_before.shape} → {x.shape}")
                print(f"  Min: {x.min().item():.4f}, Max: {x.max().item():.4f}")
                print(f"  NaN: {torch.isnan(x).sum().item()}, Inf: {torch.isinf(x).sum().item()}")
                
                if has_nan or has_inf:
                    print(f" PROBLÈME détecté à la couche {i}: {layer}")
                    print(f"   Input stats: min={x_before.min():.4f}, max={x_before.max():.4f}")
                    
                    # Analyser le layer spécifique
                    if hasattr(layer, '__len__'):  # Sequential
                        for j, sublayer in enumerate(layer):
                            print(f"     Sublayer {j}: {sublayer}")
                    
                    return False
                    
            except Exception as e:
                print(f" Couche {i} FAILED: {e}")
                print(f"   Layer: {layer}")
                return False
        
        latent = x
        print(f" Encodeur OK: {latent.shape}")
        
        # ÉTAPE 3: Test Vector Quantizer
        print("\n ÉTAPE 3: Test Vector Quantizer")
        try:
            quantized, vq_loss, indices = model.vq_layer(latent)
            print(f" VQ Layer OK:")
            print(f"  Quantized: {quantized.shape}")
            print(f"  VQ Loss: {vq_loss.item():.6f}")
            print(f"  Indices shape: {indices.shape}")
            print(f"  Unique codes: {torch.unique(indices).numel()}")
            
            # Vérifier VQ loss
            if torch.isnan(vq_loss) or torch.isinf(vq_loss):
                print(f" VQ Loss problématique: {vq_loss}")
                return False
                
        except Exception as e:
            print(f" VQ Layer FAILED: {e}")
            return False
        
        # ÉTAPE 4: Test décodeur étape par étape
        print("\n ÉTAPE 4: Test décodeur étape par étape")
        x = quantized
        
        for i, layer in enumerate(model.decoder):
            try:
                x_before = x.clone()
                x = layer(x)
                
                has_nan = torch.isnan(x).any()
                has_inf = torch.isinf(x).any()
                
                print(f"Decoder Layer {i}: {x_before.shape} → {x.shape}")
                print(f"  Min: {x.min().item():.4f}, Max: {x.max().item():.4f}")
                print(f"  NaN: {torch.isnan(x).sum().item()}, Inf: {torch.isinf(x).sum().item()}")
                
                if has_nan or has_inf:
                    print(f" PROBLÈME détecté à la couche décodeur {i}: {layer}")
                    return False
                    
            except Exception as e:
                print(f" Couche décodeur {i} FAILED: {e}")
                print(f"   Layer: {layer}")
                return False
        
        reconstruction = x
        print(f" Décodeur OK: {reconstruction.shape}")
        
        # ÉTAPE 5: Test loss function
        print("\n ÉTAPE 5: Test loss function")
        try:
            loss_dict = model.loss_function(reconstruction, masked_data, vq_loss, mask=mask)
            
            print(f" Loss function OK:")
            print(f"  Total Loss: {loss_dict['loss'].item():.6f}")
            print(f"  Recon Loss: {loss_dict['Reconstruction_Loss'].item():.6f}")
            print(f"  VQ Loss: {loss_dict['VQ_Loss'].item():.6f}")
            
            for key, value in loss_dict.items():
                if torch.isnan(value) or torch.isinf(value):
                    print(f" {key} problématique: {value}")
                    return False
                    
        except Exception as e:
            print(f" Loss function FAILED: {e}")
            return False
        
        print("\n === DIAGNOSTIC COMPLET: SUCCÈS ===")
        return True

def visualize_codebook_usage(model, dataloader, device='cuda'):
    """
    Visualise l'utilisation du codebook
    """
    import matplotlib.pyplot as plt
    
    stats = model.get_codebook_usage(dataloader, device)
    
    plt.figure(figsize=(12, 4))
    
    # Histogramme d'utilisation
    plt.subplot(1, 2, 1)
    plt.bar(range(len(stats['codebook_usage'])), stats['codebook_usage'])
    plt.title(f'Utilisation du Codebook\n{stats["active_codes"]}/{stats["total_codes"]} codes actifs ({stats["utilization_rate"]:.1%})')
    plt.xlabel('Index du code')
    plt.ylabel('Fréquence d\'utilisation')
    
    # Distribution des fréquences
    plt.subplot(1, 2, 2)
    usage_values = stats['codebook_usage'][stats['codebook_usage'] > 0]
    plt.hist(usage_values, bins=50, alpha=0.7)
    plt.title('Distribution des fréquences d\'utilisation')
    plt.xlabel('Fréquence d\'utilisation')
    plt.ylabel('Nombre de codes')
    
    plt.tight_layout()
    plt.show()
    
    return stats


def compare_reconstructions(model, test_data, device='cuda', num_samples=3):
    """
    Compare les reconstructions originales vs VQ-VAE
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    with torch.no_grad():
        for i in range(min(num_samples, len(test_data))):
            if isinstance(test_data[i], tuple):
                data, mask = test_data[i]
            else:
                data, mask = test_data[i], None
                
            if len(data.shape) == 3:
                data = data.unsqueeze(0)
            if mask is not None and len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
                
            data = data.to(device)
            if mask is not None:
                mask = mask.to(device)
            
            reconstruction = model.reconstruct(data, mask)
            
            # Visualisation pour chaque canal
            fig, axes = plt.subplots(2, data.shape[1], figsize=(15, 6))
            
            for channel in range(data.shape[1]):
                # Original
                axes[0, channel].imshow(data[0, channel].cpu().numpy(), cmap='viridis')
                axes[0, channel].set_title(f'Original - Canal {channel}')
                axes[0, channel].axis('off')
                
                # Reconstruction
                axes[1, channel].imshow(reconstruction[0, channel].cpu().numpy(), cmap='viridis')
                axes[1, channel].set_title(f'Reconstruction - Canal {channel}')
                axes[1, channel].axis('off')
            
            plt.suptitle(f'Comparaison Original vs VQ-VAE - Échantillon {i+1}')
            plt.tight_layout()
            plt.show()