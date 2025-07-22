import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import BoundaryNorm
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
from PIL import Image
import os
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config.config_vqvae import*

def visualize_vae_latent_space(target, model, test_maps, device, latent_dim, num_samples=3, save_folder=None):
    """
    Visualise l'espace latent spatial des modèles VAE et WAE-MMD de manière universelle.
    
    Args:
        target: Type de modèle ("VAE" ou "WAE-MMD")
        model: Modèle VAE ou WAE-MMD
        test_maps: Dataset de test (DataLoader)
        device: Device (cpu/cuda)
        latent_dim: Nombre de canaux latents (ex: 16)
        num_samples: Nombre d'échantillons à visualiser
        save_folder: Dossier pour sauvegarder les visualisations (auto-détecté si None)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import torch
    from matplotlib.colors import BoundaryNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    model_to_use = target
    is_wae_mmd = (model_to_use == "WAE-MMD")
    model_type = model_to_use
    
    print(f"🔍 Visualisation espace latent pour: {model_type}")
    
    if save_folder is None:
        save_folder = f"results_{model_type.lower().replace('-', '_')}_latent"
    
    model.eval()
    model.to(device)
    
    os.makedirs(save_folder, exist_ok=True)
    
    latent_maps = []
    
    # Collecter les représentations latentes
    with torch.no_grad():
        try:
            # Récupérer un batch du DataLoader
            batch = next(iter(test_maps))
            print("Type de batch:", type(batch))
            
            #  Gestion unifiée des formats de batch
            data_tensors = []
            mask_tensors = []
            
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                # Format standard: (data_batch, mask_batch)
                data_batch, mask_batch = batch
                
                print(f"✅ Format détecté: Tuple standard (data, mask)")
                print(f"  - Data batch shape: {data_batch.shape}")
                print(f"  - Mask batch shape: {mask_batch.shape}")
                
                # Limiter le nombre d'échantillons
                batch_size = data_batch.shape[0]
                samples_to_process = min(num_samples, batch_size)
                
                for i in range(samples_to_process):
                    data_tensors.append(data_batch[i:i+1])  # Garder dimension batch
                    mask_tensors.append(mask_batch[i:i+1])  # Garder dimension batch
            
            elif isinstance(batch, list):
                # Format liste: chaque élément peut être (data, mask) ou juste data
                print(f"✅ Format détecté: Liste d'échantillons")
                
                expected_channels = 5
                samples_to_process = min(num_samples, len(batch))
                
                for i in range(samples_to_process):
                    item = batch[i]
                    
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        # Échantillon avec masque
                        data, mask = item
                    else:
                        # Échantillon sans masque
                        data = item
                        mask = None
                    
                    # Vérification et conversion des données
                    if not torch.is_tensor(data):
                        data = torch.tensor(data)
                    
                    #  Validation stricte des dimensions
                    if len(data.shape) == 3:  # [C, H, W]
                        if data.shape[0] == expected_channels:
                            data = data.unsqueeze(0)  # [1, C, H, W]
                        else:
                            print(f"  ❌ Échantillon {i+1} ignoré: mauvais nombre de canaux ({data.shape[0]} au lieu de {expected_channels})")
                            continue
                    elif len(data.shape) == 4:  # [B, C, H, W]
                        if data.shape[1] != expected_channels:
                            print(f"  ❌ Échantillon {i+1} ignoré: mauvais nombre de canaux ({data.shape[1]} au lieu de {expected_channels})")
                            continue
                    else:
                        print(f"  ❌ Échantillon {i+1} ignoré: dimensions incorrectes {data.shape}")
                        continue
                    
                    # Gestion du masque
                    if mask is not None:
                        if not torch.is_tensor(mask):
                            mask = torch.tensor(mask)
                        if len(mask.shape) == 2:  # [H, W] -> [1, H, W]
                            mask = mask.unsqueeze(0)
                        if len(mask.shape) == 3:  # [1, H, W] -> [1, 1, H, W] ou [B, H, W] -> [B, 1, H, W]
                            mask = mask.unsqueeze(1)
                    
                    data_tensors.append(data)
                    mask_tensors.append(mask)
                    print(f"  ✅ Échantillon {i+1} valide: shape = {data.shape}")
            
            else:
                raise ValueError(f"Format de batch non supporté: {type(batch)}")
            
            #  Extraction des représentations latentes avec gestion d'erreurs robuste
            print(f"\n📊 Extraction des représentations latentes pour {len(data_tensors)} échantillons...")
            
            for i, (data, mask) in enumerate(zip(data_tensors, mask_tensors)):
                try:
                    print(f"Traitement échantillon {i+1}:")
                    print(f"  - Data shape: {data.shape}")
                    if mask is not None:
                        print(f"  - Mask shape: {mask.shape}")
                    
                    # Déplacer vers le device
                    data = data.to(device)
                    if mask is not None:
                        mask = mask.to(device)
                    
                    # Extraction selon le type de modèle
                    if is_wae_mmd:
                        if hasattr(model, 'encode'):
                            z = model.encode(data, mask)
                            if isinstance(z, tuple):
                                z = z[0]
                        else:
                            outputs = model(data, mask)
                            if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                                z = outputs[1]
                            else:
                                raise ValueError("Impossible d'extraire la représentation latente du WAE-MMD")
                    else:
                        # VAE: méthode standard
                        mu, logvar, _ = model.encode(data, mask)
                        z = model.reparameterize(mu, logvar)
                    
                    # Validation et normalisation des dimensions latentes
                    print(f"  - Représentation latente brute shape: {z.shape}")
                    
                    # Retirer la dimension batch si présente
                    if len(z.shape) == 4 and z.shape[0] == 1:  # [1, C, H, W] -> [C, H, W]
                        z = z.squeeze(0)
                    elif len(z.shape) == 4 and z.shape[0] > 1:  # [B, C, H, W] -> prendre le premier
                        z = z[0]
                    
                    # Vérifier que les dimensions sont correctes
                    if len(z.shape) != 3:
                        raise ValueError(f"Dimensions latentes incorrectes après traitement: {z.shape}. Attendu: [C, H, W]")
                    
                    if z.shape[0] != latent_dim:
                        print(f"   Attention: nombre de canaux latents ({z.shape[0]}) différent du paramètre latent_dim ({latent_dim})")
                        # Ajuster latent_dim si nécessaire
                        latent_dim = z.shape[0]
                    
                    latent_maps.append(z.cpu().numpy())
                    print(f"  ✅ Représentation latente extraite: shape = {z.shape}")
                    
                except Exception as e:
                    print(f"  ❌ Erreur lors de l'extraction pour l'échantillon {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
        except Exception as e:
            print("⚠️ Erreur lors de l'extraction des représentations latentes:")
            print(f"Erreur: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Vérifier qu'on a des données latentes
    if not latent_maps:
        print("❌ Aucune représentation latente extraite")
        return None
    
    print(f"\n🎯 {len(latent_maps)} représentations latentes extraites avec succès")
    
    # Visualisation robuste avec validation des données ET ÉCHELLE CORRECTE
    saved_figures = []
    
    for i, z_map in enumerate(latent_maps):
        print(f"\n📊 Visualisation échantillon {i+1}")
        print(f"  - Shape: {z_map.shape}")
        print(f"  - Canaux: {z_map.shape[0]}")
        print(f"  - Dimensions spatiales: {z_map.shape[1]} × {z_map.shape[2]}")
        
        # Validation finale des données
        if len(z_map.shape) != 3:
            print(f"  ❌ Shape incorrecte: {z_map.shape}. Ignoré.")
            continue
        
        current_latent_dim = z_map.shape[0]
        
        # Ajuster le nombre de sous-graphiques si trop élevé
        max_channels_per_figure = 20
        if current_latent_dim > max_channels_per_figure:
            print(f"  ⚠️ Trop de canaux ({current_latent_dim}), limitation à {max_channels_per_figure} pour la visualisation")
            channels_to_show = max_channels_per_figure
        else:
            channels_to_show = current_latent_dim
        
        # Reproduction exacte du système de référence
        # Utiliser les dimensions réelles mais forcer le ratio 43/90 pour l'affichage
        display_height = 43  # Hauteur d'affichage standard
        display_width = 90   # Largeur d'affichage standard
        
        map_ratio = display_height / display_width  # 43/90
        single_width = 12  # Largeur d'une carte en pouces
        single_height = single_width * map_ratio
        total_height = single_height * channels_to_show  # IMPORTANT: pas de limitation ici
        
        try:
            fig, axes = plt.subplots(channels_to_show, 1, 
                                   figsize=(single_width, total_height))
            
            if channels_to_show == 1:
                axes = [axes]
            
            for channel in range(channels_to_show):
                ax = axes[channel]
                data = z_map[channel]
                
                # Vérification des données
                if np.all(np.isnan(data)):
                    print(f"    ⚠️ Canal {channel}: toutes les valeurs sont NaN")
                    ax.text(0.5, 0.5, f'Canal {channel}: Données manquantes', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Configuration des limites avec gestion robuste
                valid_data = data[~np.isnan(data)]
                if len(valid_data) == 0:
                    vmin, vmax = -1, 1
                else:
                    vmin, vmax = np.min(valid_data), np.max(valid_data)
                    
                    if np.isnan(vmin) or np.isnan(vmax):
                        vmin, vmax = -1, 1
                    elif abs(vmax - vmin) < 1e-10:
                        range_center = (vmax + vmin) / 2
                        vmin = range_center - 0.1
                        vmax = range_center + 0.1
                
                im = ax.imshow(data, cmap='bwr', vmin=vmin, vmax=vmax,
                              aspect='auto', origin='lower',
                              extent=[0, display_width, 0, display_height])
                
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = plt.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=8)
                
                # Titres et labels comme dans la référence
                ax.set_title(f'Canal latent {channel}', fontsize=12)
                ax.set_xlabel('Longueur (90)', fontsize=10)
                ax.set_ylabel('Hauteur (43)', fontsize=10)
                ax.tick_params(axis='both', which='major', labelsize=8)
            
            plt.suptitle(f'Espace latent {model_type} - Échantillon {i+1} (Forme: {current_latent_dim}×{display_height}×{display_width})', 
                        fontsize=14, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Sauvegarde
            latent_filename = f"{save_folder}/{model_type.lower().replace('-', '_')}_latent_space_sample_{i+1}.png"
            plt.savefig(latent_filename, dpi=300, bbox_inches='tight', facecolor='white')
            saved_figures.append(latent_filename)
            print(f"  💾 Sauvegardé: {latent_filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"  ❌ Erreur lors de la visualisation de l'échantillon {i+1}: {e}")
            continue
    
    # Analyse de la variance (code existant inchangé mais avec validation)
    if latent_maps:
        print(f"\n📊 Analyse de la variance par canal latent...")
        
        # Utiliser les vraies dimensions du premier échantillon
        actual_latent_dim = latent_maps[0].shape[0]
        
        all_variances = []
        for z_map in latent_maps:
            if z_map.shape[0] != actual_latent_dim:
                print(f"⚠️ Dimensions incohérentes détectées, ignoré un échantillon")
                continue
            variances_per_channel = [np.var(z_map[channel]) for channel in range(actual_latent_dim)]
            all_variances.append(variances_per_channel)
        
        if all_variances:
            avg_variances = np.mean(all_variances, axis=0)
            
            # Graphique des variances
            plt.figure(figsize=(max(8, actual_latent_dim * 0.4), 6))
            bars = plt.bar(range(actual_latent_dim), avg_variances, alpha=0.7, color='steelblue')
            plt.axhline(y=np.mean(avg_variances), color='red', linestyle='--', 
                        label=f'Variance moyenne: {np.mean(avg_variances):.4f}')
            
            threshold = 0.01
            for i, bar in enumerate(bars):
                if avg_variances[i] > threshold:
                    bar.set_color('steelblue')
                else:
                    bar.set_color('lightcoral')
            
            plt.title(f'Variance spatiale par canal latent {model_type} ({actual_latent_dim} canaux)')
            plt.xlabel('Canal latent')
            plt.ylabel('Variance spatiale')
            plt.xticks(range(actual_latent_dim))
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend()
            plt.tight_layout()
            
            variance_filename = f"{save_folder}/{model_type.lower().replace('-', '_')}_latent_variance_analysis.png"
            plt.savefig(variance_filename, dpi=300, bbox_inches='tight', facecolor='white')
            saved_figures.append(variance_filename)
            print(f"💾 Sauvegardé: {variance_filename}")
            
            plt.show()
            
            # Statistiques avec échelle correcte basée sur vos données
            print(f"\n📈 Statistiques espace latent {model_type}:")
            first_map = latent_maps[0]
            print(f"   - Forme réelle: {actual_latent_dim} × {first_map.shape[1]} × {first_map.shape[2]}")
            print(f"   - Variance spatiale moyenne: {np.mean(avg_variances):.6f}")
            print(f"   - Variance min: {np.min(avg_variances):.6f} (canal {np.argmin(avg_variances)})")
            print(f"   - Variance max: {np.max(avg_variances):.6f} (canal {np.argmax(avg_variances)})")
            print(f"   - Canaux actifs (>0.01): {np.sum(avg_variances > 0.01)}/{actual_latent_dim}")
            
            return avg_variances
        else:
            print("❌ Impossible de calculer les variances")
            return None
    
    print(f"\n📁 Visualisations sauvegardées dans {save_folder}/")
    return None
    
def visualize_vqvae_latent_space(model, test_maps, device, num_samples=3, save_folder=None):
    """
    Visualise l'espace latent spatial du modèle VQ-VAE de manière complète.
    VERSION UNIFIÉE avec gestion robuste des batches comme visualize_vae_latent_space.
    
    Args:
        model: Modèle VQ-VAE
        test_maps: Dataset de test (DataLoader)
        device: Device (cpu/cuda)
        num_samples: Nombre d'échantillons à visualiser
        save_folder: Dossier pour sauvegarder les visualisations (auto-détecté si None)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import torch
    from matplotlib.colors import BoundaryNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    print(f"🔍 Visualisation espace latent VQ-VAE")
    
    if save_folder is None:
        save_folder = "results_vqvae_latent"
    
    model.eval()
    model.to(device)
    
    os.makedirs(save_folder, exist_ok=True)
    
    # Récupérer les dimensions du modèle VQ-VAE
    embedding_dim = model.embedding_dim
    try:
        latent_height = model.latent_height
        latent_width = model.latent_width
        num_embeddings = model.num_embeddings
    except:
        # Valeurs par défaut si pas disponibles
        latent_height = 43
        latent_width = 90
        num_embeddings = getattr(model, 'num_embeddings', 512)
    
    print(f"   Dimensions: {embedding_dim}×{latent_height}×{latent_width}")
    print(f"   Codebook: {num_embeddings} codes")
    
    latent_maps = []
    encoding_maps = []
    continuous_maps = []
    vq_losses = []
    
    # Fonction de nettoyage robuste (adaptée du VAE)
    def clean_data(data, method='clamp'):
        """Nettoie les données avec plusieurs méthodes disponibles"""
        if method == 'clamp':
            data_clean = torch.where(torch.isnan(data) | torch.isinf(data), 
                                   torch.zeros_like(data), data)
            data_clean = torch.clamp(data_clean, -3, 3)
        elif method == 'normalize':
            data_clean = torch.where(torch.isnan(data) | torch.isinf(data), 
                                   torch.zeros_like(data), data)
            p1, p99 = torch.quantile(data_clean, 0.01), torch.quantile(data_clean, 0.99)
            data_clean = torch.clamp(data_clean, p1, p99)
        else:
            data_clean = torch.where(torch.isnan(data) | torch.isinf(data), 
                                   torch.zeros_like(data), data)
        return data_clean
    
    # Extraction des représentations VQ-VAE
    def extract_vqvae_representation(data, mask=None):
        """Extrait les représentations latentes VQ-VAE de manière robuste"""
        try:
            # Étape 1: Nettoyage initial
            data_clean = clean_data(data, method='clamp')
            
            # Étape 2: Application du masque si disponible
            if hasattr(model, 'apply_mask') and mask is not None:
                clean_input, _ = model.apply_mask(data_clean, mask)
            else:
                clean_input = data_clean
            
            # Étape 3: Encodage
            latent_continuous = model.encoder(clean_input)
            
            # Vérification et nettoyage des latents continus
            if torch.isnan(latent_continuous).any() or torch.isinf(latent_continuous).any():
                print(f"    ⚠️ NaN/Inf détectés dans latent_continuous")
                latent_continuous = clean_data(latent_continuous)
            
            # Étape 4: Quantification VQ-VAE
            quantized_latents, vq_loss, quantized_indices = model.vq_layer(latent_continuous)
            
            # Vérification finale
            if torch.isnan(quantized_latents).any() or torch.isinf(quantized_latents).any():
                print(f"    ⚠️ NaN/Inf détectés dans quantized_latents")
                quantized_latents = clean_data(quantized_latents)
            
            return quantized_latents, quantized_indices, vq_loss, latent_continuous
            
        except Exception as e:
            print(f"    ❌ Erreur lors de l'extraction VQ-VAE: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None
    
    # Gestion unifiée des formats de batch (copié du VAE)
    with torch.no_grad():
        try:
            # Récupérer un batch du DataLoader
            batch = next(iter(test_maps))
            print("Type de batch:", type(batch))
            
            data_tensors = []
            mask_tensors = []
            
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                # Format standard: (data_batch, mask_batch)
                data_batch, mask_batch = batch
                
                print(f"✅ Format détecté: Tuple standard (data, mask)")
                print(f"  - Data batch shape: {data_batch.shape}")
                print(f"  - Mask batch shape: {mask_batch.shape}")
                
                # Limiter le nombre d'échantillons
                batch_size = data_batch.shape[0]
                samples_to_process = min(num_samples, batch_size)
                
                for i in range(samples_to_process):
                    data_tensors.append(data_batch[i:i+1])  # Garder dimension batch
                    mask_tensors.append(mask_batch[i:i+1])  # Garder dimension batch
            
            elif isinstance(batch, list):
                # Format liste: chaque élément peut être (data, mask) ou juste data
                print(f"✅ Format détecté: Liste d'échantillons")
                
                expected_channels = 5
                samples_to_process = min(num_samples, len(batch))
                
                for i in range(samples_to_process):
                    item = batch[i]
                    
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        # Échantillon avec masque
                        data, mask = item
                    else:
                        # Échantillon sans masque
                        data = item
                        mask = None
                    
                    # Vérification et conversion des données
                    if not torch.is_tensor(data):
                        data = torch.tensor(data)
                    
                    # Validation stricte des dimensions
                    if len(data.shape) == 3:  # [C, H, W]
                        if data.shape[0] == expected_channels:
                            data = data.unsqueeze(0)  # [1, C, H, W]
                        else:
                            print(f"  ❌ Échantillon {i+1} ignoré: mauvais nombre de canaux ({data.shape[0]} au lieu de {expected_channels})")
                            continue
                    elif len(data.shape) == 4:  # [B, C, H, W]
                        if data.shape[1] != expected_channels:
                            print(f"  ❌ Échantillon {i+1} ignoré: mauvais nombre de canaux ({data.shape[1]} au lieu de {expected_channels})")
                            continue
                    else:
                        print(f"  ❌ Échantillon {i+1} ignoré: dimensions incorrectes {data.shape}")
                        continue
                    
                    # Gestion du masque
                    if mask is not None:
                        if not torch.is_tensor(mask):
                            mask = torch.tensor(mask)
                        if len(mask.shape) == 2:  # [H, W] -> [1, H, W]
                            mask = mask.unsqueeze(0)
                        if len(mask.shape) == 3:  # [1, H, W] -> [1, 1, H, W] ou [B, H, W] -> [B, 1, H, W]
                            mask = mask.unsqueeze(1)
                    
                    data_tensors.append(data)
                    mask_tensors.append(mask)
                    print(f"  ✅ Échantillon {i+1} valide: shape = {data.shape}")
            
            else:
                raise ValueError(f"Format de batch non supporté: {type(batch)}")
            
            # EXTRACTION DES REPRÉSENTATIONS VQ-VAE
            print(f"\n📊 Extraction des représentations VQ-VAE pour {len(data_tensors)} échantillons...")
            
            for i, (data, mask) in enumerate(zip(data_tensors, mask_tensors)):
                try:
                    print(f"Traitement échantillon {i+1}:")
                    print(f"  - Data shape: {data.shape}")
                    if mask is not None:
                        print(f"  - Mask shape: {mask.shape}")
                    
                    # Déplacer vers le device
                    data = data.to(device)
                    if mask is not None:
                        mask = mask.to(device)
                    
                    # Extraction VQ-VAE
                    result = extract_vqvae_representation(data, mask)
                    if result[0] is not None:
                        q_latents, q_indices, vq_loss, cont_latents = result
                        
                        # Retirer la dimension batch si présente
                        if len(q_latents.shape) == 4 and q_latents.shape[0] == 1:
                            q_latents = q_latents.squeeze(0)
                            q_indices = q_indices.squeeze(0)
                            cont_latents = cont_latents.squeeze(0)
                        elif len(q_latents.shape) == 4 and q_latents.shape[0] > 1:
                            q_latents = q_latents[0]
                            q_indices = q_indices[0]
                            cont_latents = cont_latents[0]
                        
                        # Vérifier que les dimensions sont correctes
                        if len(q_latents.shape) != 3:
                            raise ValueError(f"Dimensions VQ-VAE incorrectes après traitement: {q_latents.shape}. Attendu: [C, H, W]")
                        
                        latent_maps.append(q_latents.cpu().numpy())
                        encoding_maps.append(q_indices.cpu().numpy())
                        continuous_maps.append(cont_latents.cpu().numpy())
                        vq_losses.append(vq_loss.item())
                        print(f"  ✅ Représentation VQ-VAE extraite: shape = {q_latents.shape}")
                        
                except Exception as e:
                    print(f"  ❌ Erreur lors de l'extraction pour l'échantillon {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
        except Exception as e:
            print("⚠️ Erreur lors de l'extraction des représentations VQ-VAE:")
            print(f"Erreur: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    # Vérifier qu'on a des données latentes
    if not latent_maps:
        print("❌ Aucune représentation VQ-VAE extraite")
        return None, None
    
    print(f"\n🎯 {len(latent_maps)} représentations VQ-VAE extraites avec succès")
    print(f"📊 VQ Loss moyenne: {np.mean(vq_losses):.6f}")
    
    # Visualisation optimisée avec échelle correcte (spécifique VQ-VAE)
    saved_figures = []
    
    for i, (z_map, enc_map, cont_map) in enumerate(zip(latent_maps, encoding_maps, continuous_maps)):
        print(f"\n📊 Visualisation échantillon {i+1}")
        print(f"  - Shape: {z_map.shape}")
        print(f"  - Canaux: {z_map.shape[0]}")
        print(f"  - Dimensions spatiales: {z_map.shape[1]} × {z_map.shape[2]}")
        
        # Validation finale des données
        if len(z_map.shape) != 3:
            print(f"  ❌ Shape incorrecte: {z_map.shape}. Ignoré.")
            continue
        
        # Vérification finale des données
        if np.isnan(z_map).any() or np.isinf(z_map).any():
            print(f"    ⚠️ Nettoyage final requis")
            z_map = np.where(np.isnan(z_map) | np.isinf(z_map), 0, z_map)
        
        current_embedding_dim = z_map.shape[0]
        
        # Paramètres de visualisation avec échelle correcte (comme la référence VQ-VAE)
        display_height = 43  # Hauteur d'affichage standard
        display_width = 90   # Largeur d'affichage standard
        
        map_ratio = display_height / display_width  # 43/90
        single_width = 12  # Largeur d'une carte en pouces
        single_height = single_width * map_ratio
        total_height = single_height * current_embedding_dim  # Pas de limitation
        
        try:
            # 1. ESPACE LATENT QUANTIFIÉ
            fig, axes = plt.subplots(current_embedding_dim, 1, 
                                   figsize=(single_width, total_height))
            
            if current_embedding_dim == 1:
                axes = [axes]  # Pour uniformiser le traitement
            
            for channel in range(current_embedding_dim):
                ax = axes[channel]
                data = z_map[channel]
                
                # Configuration des limites
                vmin, vmax = np.min(data), np.max(data)
                
                # Vérification des limites
                if np.isnan(vmin) or np.isnan(vmax) or vmax <= vmin:
                    vmin, vmax = -1, 1
                    print(f"⚠️ Limites problématiques pour canal {channel}, utilisation par défaut")
                
                if abs(vmax - vmin) < 1e-10:
                    range_center = (vmax + vmin) / 2
                    vmin = range_center - 0.1
                    vmax = range_center + 0.1
                
                im = ax.imshow(data, cmap='bwr', vmin=vmin, vmax=vmax,
                              aspect='auto', origin='lower',
                              extent=[0, display_width, 0, display_height])
                
                # Créer une colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = plt.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=8)
                
                # Configuration des axes
                ax.set_title(f'Canal latent {channel}', fontsize=12)
                ax.set_xlabel('Longueur (90)', fontsize=10)
                ax.set_ylabel('Hauteur (43)', fontsize=10)
                ax.tick_params(axis='both', which='major', labelsize=8)
            
            plt.suptitle(f'Espace latent VQ-VAE quantifié - Échantillon {i+1} (Forme: {current_embedding_dim}×{display_height}×{display_width})', 
                        fontsize=14, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Sauvegarder la figure
            latent_filename = f"{save_folder}/vqvae_quantized_latent_sample_{i+1}.png"
            plt.savefig(latent_filename, dpi=300, bbox_inches='tight', facecolor='white')
            saved_figures.append(latent_filename)
            print(f"💾 Sauvegardé: {latent_filename}")
            
            plt.show()
            
            # 2. ESPACE LATENT CONTINU (avant quantification)
            fig, axes = plt.subplots(current_embedding_dim, 1, 
                                   figsize=(single_width, total_height))
            
            if current_embedding_dim == 1:
                axes = [axes]  # Pour uniformiser le traitement
            
            for channel in range(current_embedding_dim):
                ax = axes[channel]
                data = cont_map[channel]
                
                # Configuration des limites
                vmin, vmax = np.min(data), np.max(data)
                
                # Vérification des limites
                if np.isnan(vmin) or np.isnan(vmax) or vmax <= vmin:
                    vmin, vmax = -1, 1
                    print(f"⚠️ Limites problématiques pour canal {channel}, utilisation par défaut")
                
                if abs(vmax - vmin) < 1e-10:
                    range_center = (vmax + vmin) / 2
                    vmin = range_center - 0.1
                    vmax = range_center + 0.1
                
                im = ax.imshow(data, cmap='bwr', vmin=vmin, vmax=vmax,
                              aspect='auto', origin='lower',
                              extent=[0, display_width, 0, display_height])
                
                # Créer une colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = plt.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=8)
                
                # Configuration des axes
                ax.set_title(f'Canal latent {channel}', fontsize=12)
                ax.set_xlabel('Longueur (90)', fontsize=10)
                ax.set_ylabel('Hauteur (43)', fontsize=10)
                ax.tick_params(axis='both', which='major', labelsize=8)
            
            plt.suptitle(f'Espace latent VQ-VAE continu - Échantillon {i+1} (Forme: {current_embedding_dim}×{display_height}×{display_width})', 
                        fontsize=14, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Sauvegarder la figure
            latent_filename = f"{save_folder}/vqvae_continuous_latent_sample_{i+1}.png"
            plt.savefig(latent_filename, dpi=300, bbox_inches='tight', facecolor='white')
            saved_figures.append(latent_filename)
            print(f"💾 Sauvegardé: {latent_filename}")
            
            plt.show()
            
            # 3. Analyse du codebook (spécifique VQ-VAE)
            plt.figure(figsize=(15, 5))
            
            # Carte des indices avec échelle correcte
            plt.subplot(1, 3, 1)
            im_enc = plt.imshow(enc_map, cmap='tab20', aspect='auto', origin='lower',
                               extent=[0, display_width, 0, display_height])
            plt.title(f'Indices du Codebook')
            plt.xlabel('Longueur (90)')
            plt.ylabel('Hauteur (43)')
            plt.colorbar(im_enc, fraction=0.046, pad=0.04)
            
            # Histogramme d'utilisation
            plt.subplot(1, 3, 2)
            unique_codes, counts = np.unique(enc_map, return_counts=True)
            plt.bar(unique_codes, counts, alpha=0.7, color='steelblue')
            plt.title(f'Utilisation des codes\n{len(unique_codes)}/{num_embeddings} actifs')
            plt.xlabel('Index du code')
            plt.ylabel('Fréquence')
            plt.grid(True, alpha=0.3)
            
            # Statistiques globales
            plt.subplot(1, 3, 3)
            if len(encoding_maps) > 1:
                # Analyse de la cohérence entre échantillons
                common_codes = set(encoding_maps[0].flatten())
                for enc in encoding_maps[1:]:
                    common_codes &= set(enc.flatten())
                
                plt.text(0.5, 0.5, f'Codes communs:\n{len(common_codes)}/{num_embeddings}',
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            else:
                plt.text(0.5, 0.5, f'Codes utilisés:\n{len(unique_codes)}/{num_embeddings}',
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            plt.title('Statistiques')
            
            plt.tight_layout()
            
            filename = f"{save_folder}/vqvae_codebook_analysis_sample_{i+1}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            saved_figures.append(filename)
            plt.show()
            
        except Exception as e:
            print(f"  ❌ Erreur lors de la visualisation de l'échantillon {i+1}: {e}")
            continue
    
    # Analyse statistique finale (spécifique VQ-VAE)
    if latent_maps:
        print(f"\n📊 Résumé de l'analyse VQ-VAE:")
        print(f"   - Échantillons traités: {len(latent_maps)}")
        print(f"   - VQ Loss moyenne: {np.mean(vq_losses):.6f}")
        print(f"   - Variance latente moyenne: {np.mean([np.var(z) for z in latent_maps]):.6f}")
        print(f"   - Codes uniques utilisés: {len(set().union(*[set(enc.flatten()) for enc in encoding_maps]))}")
        print(f"   - Figures sauvegardées: {len(saved_figures)}")
        print(f"   - Forme réelle: {latent_maps[0].shape[0]} × {latent_maps[0].shape[1]} × {latent_maps[0].shape[2]}")
    
    print(f"\n📁 Visualisations sauvegardées dans {save_folder}/")
    return latent_maps, encoding_maps


def adaptive_visualize_latent_vectors(target, model, test_maps, device, latent_dim, num_samples=3):
    
    """
    Fonction adaptative qui détecte le type de modèle et utilise la bonne visualisation
    
    Args:
        model: Modèle VAE ou VQ-VAE
        test_maps: Dataset de test
        device: Device
        latent_dim: Dimension latente (pour VAE uniquement)
        num_samples: Nombre d'échantillons
    """
    # Détection du type de modèle
    is_vqvae = hasattr(model, 'vq_layer') and hasattr(model, 'num_embeddings')
    
    if is_vqvae:
        print("🔍 Détection: Modèle VQ-VAE - Utilisation de la visualisation VQ-VAE")
        return visualize_vqvae_latent_space(model, test_maps, device, num_samples)
    else:
        print("🔍 Détection: Modèle VAE classique - Utilisation de la visualisation VAE")
        if latent_dim is None:
            raise ValueError("latent_dim requis pour la visualisation VAE")
        return visualize_vae_latent_space(target, model, test_maps, device, latent_dim, num_samples)
        

def visualize_reconstruction_universal(target, model, test_maps, stats_dict, device, variable_names, ds, time_indices, save_folder=None):
    """
    Visualise les reconstructions de données océaniques pour VAE ou VQ-VAE.
    Utilise la variable globale model_to_use pour déterminer le type de modèle.
    Crée des vidéos AVI au lieu de MP4 pour un meilleur support du zoom.
    
    Paramètres:
    -----------
    model : nn.Module
        Modèle VAE ou VQ-VAE à évaluer
    test_maps : list
        Liste de tuples (data, mask)
    stats_dict : dict
        Dictionnaire des statistiques pour chaque variable
    device : torch.device
        Appareil sur lequel effectuer les calculs
    variable_names : list
        Liste des noms des variables utilisées
    ds : xarray.Dataset
        Dataset original contenant les données océaniques
    time_indices : list, optional
        Liste des indices temporels correspondant aux données de test
    save_folder : str, optional
        Dossier pour sauvegarder les reconstructions (auto-détecté si None)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import cv2
    from matplotlib.colors import BoundaryNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.animation as animation
    from PIL import Image
    import torch
    
    # Utilisation de la variable globale model_to_use pour identifier de quel modèle on doit visualiser les résultats
    model_to_use = target
    is_vqvae = (model_to_use == "VQVAE")
    is_wae_mmd = (model_to_use == "WAE-MMD")
    model_type = model_to_use  # Utilise directement la variable globale
    
    print(f"🔍 Type de modèle utilisé: {model_type}")
    
    # Dossier de sauvegarde automatique si non spécifié
    if save_folder is None:
        save_folder = f"results_{model_type.lower().replace('-', '_')}"
    
    model.eval()
    model.to(device)
    
    epsilon = 1e-8
    
    # Créer le dossier de sauvegarde
    os.makedirs(save_folder, exist_ok=True)
    
    # Liste pour stocker toutes les figures à sauvegarder
    all_figures = []
    
    # CORRECTION: Compter le nombre total d'échantillons dans tous les batches
    total_samples = 0
    
    # Fonction pour compter les échantillons selon le format
    def count_total_samples(test_maps):
        """Compte le nombre total d'échantillons dans le dataset/dataloader"""
        sample_count = 0
        
        try:
            # Cas 1: DataLoader PyTorch
            if hasattr(test_maps, '__iter__') and hasattr(test_maps, 'dataset'):
                print("🔍 Format détecté: DataLoader PyTorch")
                try:
                    # Méthode directe si disponible
                    if hasattr(test_maps.dataset, '__len__'):
                        sample_count = len(test_maps.dataset)
                        print(f"   - Taille dataset: {sample_count}")
                    else:
                        # Parcourir pour compter
                        for batch in test_maps:
                            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                                data_batch = batch[0]
                                if hasattr(data_batch, 'shape') and len(data_batch.shape) >= 1:
                                    sample_count += data_batch.shape[0]
                            elif hasattr(batch, 'shape') and len(batch.shape) >= 1:
                                sample_count += batch.shape[0]
                        print(f"   - Échantillons comptés par parcours: {sample_count}")
                except Exception as e:
                    print(f"   ⚠️ Erreur comptage DataLoader: {e}")
                    sample_count = 0
                    
            # Cas 2: Liste de batches
            elif isinstance(test_maps, list):
                print("🔍 Format détecté: Liste")
                
                # Sous-cas: liste de tuples (data_batch, mask_batch)
                if len(test_maps) > 0 and isinstance(test_maps[0], (list, tuple)) and len(test_maps[0]) == 2:
                    # Vérifier si c'est des batches ou des échantillons individuels
                    first_data = test_maps[0][0]
                    if hasattr(first_data, 'shape') and len(first_data.shape) == 4:  # [B, C, H, W]
                        print("   - Détecté: Liste de batches avec shape [B, C, H, W]")
                        for data_batch, _ in test_maps:
                            if hasattr(data_batch, 'shape') and len(data_batch.shape) >= 1:
                                sample_count += data_batch.shape[0]
                    else:
                        print("   - Détecté: Liste d'échantillons individuels")
                        sample_count = len(test_maps)
                        
                # Sous-cas: liste d'échantillons individuels
                else:
                    print("   - Détecté: Liste d'échantillons")
                    sample_count = len(test_maps)
                    
                print(f"   - Total échantillons: {sample_count}")
                    
            # Cas 3: Itérateur générique
            else:
                print("🔍 Format détecté: Itérateur générique")
                try:
                    # Parcourir pour compter (attention: consomme l'itérateur!)
                    for batch in test_maps:
                        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                            data_batch = batch[0]
                            if hasattr(data_batch, 'shape') and len(data_batch.shape) >= 1:
                                sample_count += data_batch.shape[0]
                        elif hasattr(batch, 'shape') and len(batch.shape) >= 1:
                            sample_count += batch.shape[0]
                    print(f"   - Échantillons comptés: {sample_count}")
                except Exception as e:
                    print(f"   ⚠️ Erreur comptage itérateur: {e}")
                    sample_count = 0
                    
        except Exception as e:
            print(f"❌ Erreur lors du comptage des échantillons: {e}")
            sample_count = 0
            
        return sample_count
    
    # Compter les échantillons totaux
    total_samples = count_total_samples(test_maps)
    
    # Données pour les AVI (si on a assez d'échantillons)
    avi_data = {var: {'originals': [], 'reconstructions': [], 'errors': [], 'codes': [], 'masks': []} 
                for var in variable_names}
    
    # Détection basée sur le nombre total d'échantillons
    create_avis = total_samples >= 10
    
    print(f"📊 Échantillons totaux détectés: {total_samples}")
    print(f"🎬 Création de vidéos AVI: {'✅ OUI' if create_avis else '❌ NON'} (minimum: 10)")
    
    # Traitement séparé pour visualisations statiques vs AVI
    
    # 1. Visualisation statiques (limitées à 5 échantillons)
    num_samples_static = min(5, total_samples)
    static_processed = 0
    
    # 2. Collecte pour AVI (tous les échantillons si create_avis=True)
    avi_samples_to_collect = min(200, total_samples) if create_avis else 0  # Limite raisonnable
    avi_collected = 0
    
    print(f"📊 Plan de traitement:")
    print(f"   - Visualisations statiques: {num_samples_static} échantillons")
    print(f"   - Collecte pour AVI: {avi_samples_to_collect} échantillons")
    
    # Traitement des données
    try:
        batch_num = 0
        for batch in test_maps:
            batch_num += 1
            print(f"\n🔄 Traitement batch {batch_num}...")
            
            # Gérer les différents formats de batch
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data_batch, mask_batch = batch
            else:
                data_batch = batch
                mask_batch = None
            
            # S'assurer que c'est des tenseurs
            if not torch.is_tensor(data_batch):
                data_batch = torch.tensor(data_batch)
            if mask_batch is not None and not torch.is_tensor(mask_batch):
                mask_batch = torch.tensor(mask_batch)
            
            batch_size = data_batch.shape[0]
            print(f"   - Taille du batch: {batch_size}")
            
            # Traitement de chaque échantillon dans le batch
            for sample_idx in range(batch_size):
                # Extraire l'échantillon
                sample_data = data_batch[sample_idx:sample_idx+1].to(device)
                sample_mask = mask_batch[sample_idx:sample_idx+1].to(device) if mask_batch is not None else None
                
                global_sample_idx = (batch_num - 1) * batch_size + sample_idx
                
                # Visualisation statique (seulement pour les 5 premiers)
                if static_processed < num_samples_static:
                    print(f"   📊 Visualisation statique échantillon {static_processed + 1}/{num_samples_static}")
                    
                    # [Ici votre code existant pour les visualisations statiques]
                    # (creation des figures détaillées, etc.)
                    
                    static_processed += 1
                
                # CO (pour tous les échantillons si activé)
                if create_avis and avi_collected < avi_samples_to_collect:
                    try:
                        print(f"   🎬 Collecte AVI échantillon {avi_collected + 1}/{avi_samples_to_collect}")
                        
                        # Reconstruction avec le modèle
                        with torch.no_grad():
                            if is_vqvae:
                                # Pour VQ-VAE
                                if hasattr(model, 'apply_mask') and sample_mask is not None:
                                    clean_input, valid_pixels = model.apply_mask(sample_data, sample_mask)
                                else:
                                    clean_input = sample_data
                                
                                # Encodage
                                encoded = model.encoder(clean_input)
                                
                                # Quantification
                                quantized, vq_loss, encoding_indices = model.vq_layer(encoded)
                                
                                # Décodage
                                reconstructed = model.decoder(quantized)
                                
                                # Codes pour visualisation
                                codes_np = encoding_indices.squeeze().cpu().numpy()
                                
                            elif is_wae_mmd:
                                # Pour WAE-MMD
                                if hasattr(model, 'encode'):
                                    z = model.encode(sample_data, sample_mask)
                                    if isinstance(z, tuple):
                                        z = z[0]
                                else:
                                    outputs = model(sample_data, sample_mask)
                                    z = outputs[1] if isinstance(outputs, (list, tuple)) and len(outputs) >= 2 else outputs
                                
                                reconstructed = model.decode(z)
                                codes_np = None
                                
                            else:
                                # Pour VAE standard
                                mu, logvar, _ = model.encode(sample_data, sample_mask)
                                z = model.reparameterize(mu, logvar)
                                reconstructed = model.decode(z)
                                codes_np = None
                        
                        # Dénormaliser et convertir pour AVI
                        sample_np = sample_data.squeeze().cpu().numpy()
                        reconstructed_np = reconstructed.squeeze().cpu().numpy()
                        mask_np = sample_mask.squeeze().cpu().numpy() if sample_mask is not None else None
                        
                        # Dénormalisation par variable
                        for var_idx, var_name in enumerate(variable_names):
                            if var_name in stats_dict:
                                mean_val = stats_dict[var_name]['mean']
                                std_val = stats_dict[var_name]['std']
                                
                                # Dénormaliser
                                original_denorm = sample_np[var_idx] * std_val + mean_val
                                recon_denorm = reconstructed_np[var_idx] * std_val + mean_val
                                
                                # Appliquer le masque
                                if mask_np is not None:
                                    mask_2d = mask_np if len(mask_np.shape) == 2 else mask_np[0]
                                    original_denorm = np.where(mask_2d, original_denorm, np.nan)
                                    recon_denorm = np.where(mask_2d, recon_denorm, np.nan)
                                
                                # Calculer l'erreur
                                error = np.abs(original_denorm - recon_denorm)
                                
                                # Stocker pour AVI
                                avi_data[var_name]['originals'].append(original_denorm)
                                avi_data[var_name]['reconstructions'].append(recon_denorm)
                                avi_data[var_name]['errors'].append(error)
                                
                                if codes_np is not None:
                                    avi_data[var_name]['codes'].append(codes_np)
                                
                                if mask_np is not None:
                                    avi_data[var_name]['masks'].append(mask_2d)
                        
                        avi_collected += 1
                        
                        if avi_collected % 50 == 0:
                            print(f"   🎬 {avi_collected} échantillons collectés pour AVI...")
                        
                    except Exception as e:
                        print(f"   ❌ Erreur collecte AVI échantillon {global_sample_idx}: {e}")
                        continue
                
                # Arrêter si on a tout ce qu'il faut
                if static_processed >= num_samples_static and (not create_avis or avi_collected >= avi_samples_to_collect):
                    break
            
            # Arrêter si on a tout ce qu'il faut
            if static_processed >= num_samples_static and (not create_avis or avi_collected >= avi_samples_to_collect):
                break
    
    except Exception as e:
        print(f"❌ Erreur lors du traitement des batches: {e}")
        import traceback
        traceback.print_exc()
    
    # DIAGNOSTIC FINAL
    if create_avis:
        print(f"\n📊 DIAGNOSTIC COLLECTE DONNÉES AVI:")
        for var_name in variable_names:
            data = avi_data[var_name]
            print(f"\n   Variable: {var_name}")
            print(f"   - Originals: {len(data['originals'])} éléments")
            print(f"   - Reconstructions: {len(data['reconstructions'])} éléments") 
            print(f"   - Errors: {len(data['errors'])} éléments")
            print(f"   - Codes: {len(data['codes'])} éléments")
            
            if len(data['originals']) > 0:
                print(f"   - Shape des données: {data['originals'][0].shape}")
                print(f"   - Type des données: {type(data['originals'][0])}")
            
            # Vérification de cohérence
            lengths = [len(data['originals']), len(data['reconstructions']), len(data['errors'])]
            if len(set(lengths)) == 1:
                print(f"   ✅ Cohérent: {lengths[0]} frames pour tous les types")
            else:
                print(f"   ❌ Incohérent: {lengths}")
        
        print(f"\n🔍 RÉSUMÉ:")
        print(f"   - Échantillons collectés pour AVI: {avi_collected}")
        print(f"   - Échantillons traités pour visualisations: {static_processed}")
        print(f"   - Frames par variable: {len(avi_data[variable_names[0]]['originals'])}")
    
    # Limite sur le nombre d'échantillons à traiter pour les visualisations statiques
    num_samples_to_process = min(5, total_samples)
    
    # Suivi des MRE par variable
    mre_by_variable = {var: [] for var in variable_names}
    
    # Statistiques VQ-VAE
    if is_vqvae:
        vq_stats = {'total_codes_used': set(), 'vq_losses': []}
    
    # Statistiques WAE-MMD
    if is_wae_mmd:
        wae_stats = {'mmd_losses': [], 'reconstruction_losses': []}
    
    with torch.no_grad():
        num_samples_to_process = min(10 if create_avis else 5, len(test_maps))
        
        for i, (batch, ocean_mask) in enumerate(test_maps):
            # Préparation des données
            if len(batch.shape) == 3:
                batch = batch.unsqueeze(0)
            if len(ocean_mask.shape) == 2:
                ocean_mask = ocean_mask.unsqueeze(0)
                
            batch = batch.to(device)
            ocean_mask = ocean_mask.to(device)
            
            # Forward pass adapté selon le type de modèle
            if is_vqvae:
                # VQ-VAE: 4 sorties
                outputs = model(batch, ocean_mask)
                recon_batch, input_batch, vq_loss, encoding_indices = outputs
                
                # Statistiques VQ-VAE spécifiques
                unique_codes = torch.unique(encoding_indices.cpu())
                codes_used = len(unique_codes)
                vq_stats['total_codes_used'].update(unique_codes.numpy())
                vq_stats['vq_losses'].append(vq_loss.item())
                
                print(f"📊 Échantillon {i+1}: VQ loss = {vq_loss.item():.6f}, "
                      f"Codes utilisés = {codes_used}/{model.num_embeddings}")
            elif is_wae_mmd:
                # WAE-MMD: 4 sorties (recon, z, mmd_loss, recon_loss)
                outputs = model(batch, ocean_mask)
                if len(outputs) == 4:
                    recon_batch, z, mmd_loss, recon_loss = outputs
                    wae_stats['mmd_losses'].append(mmd_loss.item())
                    wae_stats['reconstruction_losses'].append(recon_loss.item())
                    print(f"📊 Échantillon {i+1}: MMD loss = {mmd_loss.item():.6f}, "
                          f"Recon loss = {recon_loss.item():.6f}")
                else:
                    # Fallback si le modèle retourne seulement la reconstruction
                    recon_batch = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                    print(f"📊 Échantillon {i+1}: WAE-MMD (sorties simplifiées)")
                
                encoding_indices = None  # Pas de codes pour WAE-MMD
                vq_loss = None
            else:
                # VAE: 3 sorties
                recon_batch, mu, logvar = model(batch, ocean_mask)
                encoding_indices = None  # Pas de codes pour VAE
                vq_loss = None
            
            # Copier les données pour la dénormalisation
            batch_denorm = batch.clone()
            recon_denorm = recon_batch.clone()
            
            # Inverser la normalisation
            for j, var in enumerate(variable_names):
                mean = stats_dict[var]['mean']
                std = stats_dict[var]['std'] * 3  # Car on a divisé par 3*std dans le preprocessing
                batch_denorm[:, j, :, :] = batch_denorm[:, j, :, :] * std + mean
                recon_denorm[:, j, :, :] = recon_denorm[:, j, :, :] * std + mean
            
            # Expand le masque pour tous les canaux
            mask_expanded = ocean_mask.unsqueeze(1).expand(batch.size(0), batch.size(1), -1, -1)
            
            # Visualisation pour chaque variable
            batch_idx = 0  # Première carte du batch
            for j, var in enumerate(variable_names):
                # Extraire les données NORMALISÉES (pour calcul MAE/MRE normalisées)
                original_data_norm = batch[batch_idx, j].cpu().numpy()
                recon_data_norm = recon_batch[batch_idx, j].cpu().numpy()
                
                # Extraire les données DÉNORMALISÉES (pour calcul MAE/MRE dénormalisées)
                original_data_denorm = batch_denorm[batch_idx, j].cpu().numpy()
                recon_data_denorm = recon_denorm[batch_idx, j].cpu().numpy()
                
                mask_np = ocean_mask[batch_idx].cpu().numpy() > 0
                
                # Créer des copies pour la visualisation avec NaN sur terre (utilise les données dénormalisées)
                original_data_masked = original_data_denorm.copy()
                recon_data_masked = recon_data_denorm.copy()
                
                # Remplacer les valeurs terrestres par NaN pour la visualisation
                original_data_masked[~mask_np] = np.nan
                recon_data_masked[~mask_np] = np.nan
                
                # ===== CALCULS DES ERREURS SUR DONNÉES NORMALISÉES =====
                abs_error_norm = np.abs(recon_data_norm - original_data_norm)
                rel_error_norm = abs_error_norm / (np.abs(original_data_norm) + epsilon)
                mean_abs_error_norm = np.mean(abs_error_norm[mask_np]) if mask_np.any() else 0
                mean_rel_error_norm = np.mean(rel_error_norm[mask_np]) if mask_np.any() else 0
                
                # ===== CALCULS DES ERREURS SUR DONNÉES DÉNORMALISÉES =====
                abs_error_denorm = np.abs(recon_data_denorm - original_data_denorm)
                rel_error_denorm = abs_error_denorm / (np.abs(original_data_denorm) + epsilon)
                mean_abs_error_denorm = np.mean(abs_error_denorm[mask_np]) if mask_np.any() else 0
                mean_rel_error_denorm = np.mean(rel_error_denorm[mask_np]) if mask_np.any() else 0
                
                # Stocker les MRE dénormalisées pour les statistiques finales (pour cohérence avec le reste du code)
                mre_by_variable[var].append(mean_rel_error_denorm)
                
                # Créer l'erreur absolue dénormalisée avec NaN sur terre (pour la visualisation)
                abs_error_masked = abs_error_denorm.copy()
                abs_error_masked[~mask_np] = np.nan
                
                # Déterminer les limites colorimétriques (utilise les données dénormalisées pour la visualisation)
                vmin = np.nanmin([np.nanmin(original_data_masked), np.nanmin(recon_data_masked)])
                vmax = np.nanmax([np.nanmax(original_data_masked), np.nanmax(recon_data_masked)])
                err_max = np.nanmax(abs_error_masked)
                
                # Vérification et ajustement des limites pour éviter les colorbars plates
                if np.isnan(vmin) or np.isnan(vmax) or vmax <= vmin:
                    print(f"⚠️ Limites colorimétriques problématiques pour {var}: vmin={vmin}, vmax={vmax}")
                    # Utiliser les données brutes comme fallback
                    vmin_orig = np.nanmin(original_data_masked)
                    vmax_orig = np.nanmax(original_data_masked)
                    vmin_recon = np.nanmin(recon_data_masked)
                    vmax_recon = np.nanmax(recon_data_masked)
                    
                    if not np.isnan(vmin_orig) and not np.isnan(vmax_orig):
                        vmin, vmax = vmin_orig, vmax_orig
                    elif not np.isnan(vmin_recon) and not np.isnan(vmax_recon):
                        vmin, vmax = vmin_recon, vmax_recon
                    else:
                        # Dernière option : utiliser une plage arbitraire
                        vmin, vmax = -1, 1
                        print(f"🔧 Utilisation de limites par défaut pour {var}: [{vmin}, {vmax}]")
                
                # S'assurer qu'il y a une différence minimale entre vmin et vmax
                if abs(vmax - vmin) < 1e-10:
                    range_center = (vmax + vmin) / 2
                    vmin = range_center - 0.1
                    vmax = range_center + 0.1
                    print(f"🔧 Ajustement de la plage pour {var}: [{vmin:.3f}, {vmax:.3f}]")
                
                # Paramètres de la figure pour respecter l'échelle géographique
                map_ratio = original_data_denorm.shape[0] / original_data_denorm.shape[1]  # hauteur/largeur
                single_width = 12  # Largeur d'une carte (augmentée)
                single_height = single_width * map_ratio
                
                # LAYOUT ADAPTÉ SELON LE TYPE DE MODÈLE
                if is_vqvae:
                    # VQ-VAE: Layout avec 4 panels (ajout carte des codes)
                    fig = plt.figure(figsize=(2 * single_width, 3 * single_height))
                    
                    ax1 = plt.subplot2grid((3, 2), (0, 0))  # Original
                    ax2 = plt.subplot2grid((3, 2), (0, 1))  # Reconstruction
                    ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)  # Erreur (toute la largeur)
                    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)  # Codes VQ (toute la largeur)
                else:
                    # VAE et WAE-MMD: Layout classique avec 3 panels
                    fig = plt.figure(figsize=(2 * single_width, 2.5 * single_height))
                    
                    ax1 = plt.subplot2grid((2, 2), (0, 0))  # Original
                    ax2 = plt.subplot2grid((2, 2), (0, 1))  # Reconstruction
                    ax3 = plt.subplot2grid((2, 1), (1, 0))  # Erreur (toute la largeur)
                    ax4 = None  # Pas de 4ème panel pour VAE et WAE-MMD
                
                # Palettes avec couleur de fond pour NaN (continents)
                cmap = plt.cm.viridis.copy()
                cmap.set_bad('lightgray')
                
                err_cmap = plt.cm.Reds.copy()
                err_cmap.set_bad('lightgray')
                
                # CRÉATION DES NORMALISATIONS CONTINUES
                # Pour les données principales (original et reconstruction)
                data_norm = plt.Normalize(vmin=vmin, vmax=vmax)

                # Pour l'erreur
                if np.isnan(err_max) or err_max <= 0:
                    err_max = 0.1
                    print(f"🔧 Erreur max problématique pour {var}, utilisation de {err_max}")

                err_norm = plt.Normalize(vmin=0, vmax=err_max)
                
                # PREMIÈRE LIGNE : Original et Reconstruction
                # Original
                im1 = ax1.imshow(original_data_masked, cmap=cmap, norm=data_norm,
                                aspect='equal', origin='lower')
                ax1.set_title(f"{var} - Original", fontsize=14)
                ax1.set_xlabel(f'Longitude ({original_data_denorm.shape[1]})', fontsize=12)
                ax1.set_ylabel(f'Latitude ({original_data_denorm.shape[0]})', fontsize=12)
                
                # Colorbar continue pour l'original
                divider1 = make_axes_locatable(ax1)
                cax1 = divider1.append_axes("right", size="5%", pad=0.1)
                cbar1 = plt.colorbar(im1, cax=cax1)
                cbar1.ax.tick_params(labelsize=10)
                
                # Reconstruction
                im2 = ax2.imshow(recon_data_masked, cmap=cmap, norm=data_norm,
                                aspect='equal', origin='lower')
                ax2.set_title(f"{var} - Reconstruit ({model_type})", fontsize=14)
                ax2.set_xlabel(f'Longitude ({original_data_denorm.shape[1]})', fontsize=12)
                ax2.set_ylabel(f'Latitude ({original_data_denorm.shape[0]})', fontsize=12)
                
                # Colorbar continue pour la reconstruction
                divider2 = make_axes_locatable(ax2)
                cax2 = divider2.append_axes("right", size="5%", pad=0.1)
                cbar2 = plt.colorbar(im2, cax=cax2)
                cbar2.ax.tick_params(labelsize=10)
                
                # DEUXIÈME LIGNE : Erreur absolue (pleine largeur) - TITRE MODIFIÉ
                im3 = ax3.imshow(abs_error_masked, cmap=err_cmap, norm=err_norm,
                                aspect='equal', origin='lower')
                
                # Titre avec les deux types d'erreurs
                error_title = (f"{var} - Erreur absolue\n"
                              f"Données dénormalisées: MAE={mean_abs_error_denorm:.4f}, MRE={mean_rel_error_denorm:.4f}\n"
                              f"Données normalisées: MAE={mean_abs_error_norm:.4f}, MRE={mean_rel_error_norm:.4f}")
                ax3.set_title(error_title, fontsize=12)
                ax3.set_xlabel(f'Longitude ({original_data_denorm.shape[1]})', fontsize=12)
                ax3.set_ylabel(f'Latitude ({original_data_denorm.shape[0]})', fontsize=12)
                
                # Colorbar continue pour l'erreur
                divider3 = make_axes_locatable(ax3)
                cax3 = divider3.append_axes("right", size="5%", pad=0.1)
                cbar3 = plt.colorbar(im3, cax=cax3)
                cbar3.ax.tick_params(labelsize=10)
                
                # TROISIÈME LIGNE : Codes VQ (seulement pour VQ-VAE)
                if is_vqvae and ax4 is not None:
                    encoding_2d = encoding_indices[batch_idx].cpu().numpy()
                    im4 = ax4.imshow(encoding_2d, cmap='tab20', aspect='equal', origin='lower')
                    ax4.set_title(f"{var} - Codes VQ utilisés ({codes_used}/{model.num_embeddings} codes actifs)", fontsize=14)
                    ax4.set_xlabel(f'Longitude ({original_data_denorm.shape[1]})', fontsize=12)
                    ax4.set_ylabel(f'Latitude ({original_data_denorm.shape[0]})', fontsize=12)
                    
                    # Colorbar pour les codes (déjà discrète par nature)
                    divider4 = make_axes_locatable(ax4)
                    cax4 = divider4.append_axes("right", size="5%", pad=0.1)
                    cbar4 = plt.colorbar(im4, cax=cax4, label='Index du code')
                    cbar4.ax.tick_params(labelsize=10)
                
                # Titre général adapté - MODIFIÉ POUR INCLURE LES DEUX TYPES D'ERREURS
                time_info = f" (t={time_indices[i]})" if time_indices is not None else ""
                if is_vqvae:
                    title = (f"Reconstruction {model_type} - {var}{time_info}\n"
                           f"Dénormalisées: MAE={mean_abs_error_denorm:.4f}, MRE={mean_rel_error_denorm:.4f} | "
                           f"Normalisées: MAE={mean_abs_error_norm:.4f}, MRE={mean_rel_error_norm:.4f}\n"
                           f"VQ Loss: {vq_loss.item():.6f} | Codes: {codes_used}/{model.num_embeddings}")
                elif is_wae_mmd:
                    if 'mmd_loss' in locals() and 'recon_loss' in locals():
                        title = (f"Reconstruction {model_type} - {var}{time_info}\n"
                               f"Dénormalisées: MAE={mean_abs_error_denorm:.4f}, MRE={mean_rel_error_denorm:.4f} | "
                               f"Normalisées: MAE={mean_abs_error_norm:.4f}, MRE={mean_rel_error_norm:.4f}\n"
                               f"MMD Loss: {mmd_loss.item():.6f} | Recon Loss: {recon_loss.item():.6f}")
                    else:
                        title = (f"Reconstruction {model_type} - {var}{time_info}\n"
                               f"Dénormalisées: MAE={mean_abs_error_denorm:.4f}, MRE={mean_rel_error_denorm:.4f} | "
                               f"Normalisées: MAE={mean_abs_error_norm:.4f}, MRE={mean_rel_error_norm:.4f}")
                else:
                    title = (f"Reconstruction {model_type} - {var}{time_info}\n"
                           f"Dénormalisées: MAE={mean_abs_error_denorm:.4f}, MRE={mean_rel_error_denorm:.4f} | "
                           f"Normalisées: MAE={mean_abs_error_norm:.4f}, MRE={mean_rel_error_norm:.4f}")
                
                plt.suptitle(title, fontsize=14, y=0.98)
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.85)  # Ajusté pour laisser plus de place au titre
                
                # Stocker les données pour les AVI (utilise les données dénormalisées pour la visualisation)
                if create_avis and i < 10:
                    avi_data[var]['originals'].append(original_data_masked.copy())
                    avi_data[var]['reconstructions'].append(recon_data_masked.copy())
                    avi_data[var]['errors'].append(abs_error_masked.copy())
                    if is_vqvae:
                        avi_data[var]['codes'].append(encoding_2d.copy())
                    if len(avi_data[var]['masks']) == 0:  # Stocker le masque une seule fois
                        avi_data[var]['masks'] = mask_np
                
                # Stocker la figure pour sauvegarde finale (seulement les 5 premiers pour PDF)
                if i < 5:
                    all_figures.append((fig, var, i))
                
                # Afficher seulement les 5 premiers échantillons
                if i < 5:
                    plt.show()
                else:
                    plt.close(fig)  # Fermer sans afficher pour économiser la mémoire
            
            if i >= (num_samples_to_process - 1):  # Traiter 10 échantillons si AVI, sinon 5
                break
    
    # STATISTIQUES FINALES DES MRE PAR VARIABLE
    print(f"\n📊 Moyennes des MRE de reconstruction par variable:")
    print(f"   (Calculées sur {num_samples_to_process} échantillons de test)")
    total_mre = 0
    valid_variables = 0
    
    for var in variable_names:
        if mre_by_variable[var]:
            avg_mre = np.mean(mre_by_variable[var])
            std_mre = np.std(mre_by_variable[var])
            min_mre = np.min(mre_by_variable[var])
            max_mre = np.max(mre_by_variable[var])
            
            print(f"   {var:>8}: MRE = {avg_mre:.4f} ± {std_mre:.4f} (min: {min_mre:.4f}, max: {max_mre:.4f})")
            total_mre += avg_mre
            valid_variables += 1
    
    if valid_variables > 0:
        global_avg_mre = total_mre / valid_variables
        print(f"   {'GLOBAL':>8}: MRE = {global_avg_mre:.4f} (moyenne de toutes les variables)")
        
        # Identifier la meilleure et pire variable
        avg_mres = {var: np.mean(mre_by_variable[var]) for var in variable_names if mre_by_variable[var]}
        if avg_mres:
            best_var = min(avg_mres, key=avg_mres.get)
            worst_var = max(avg_mres, key=avg_mres.get)
            print(f"   🏆 Meilleure: {best_var} (MRE = {avg_mres[best_var]:.4f})")
            print(f"   📉 À améliorer: {worst_var} (MRE = {avg_mres[worst_var]:.4f})")
    
    # STATISTIQUES FINALES VQ-VAE
    if is_vqvae:
        total_codes_used = len(vq_stats['total_codes_used'])
        utilization_rate = total_codes_used / model.num_embeddings
        avg_vq_loss = np.mean(vq_stats['vq_losses'])
        
        print(f"\n📊 Statistiques {model_type}:")
        print(f"   Utilisation du codebook: {utilization_rate:.1%} ({total_codes_used}/{model.num_embeddings})")
        print(f"   VQ Loss moyenne: {avg_vq_loss:.6f}")
        
        if utilization_rate < 0.3:
            print(f"💡 Codebook sous-utilisé - considérer réduire num_embeddings")
        elif utilization_rate > 0.8:
            print(f"✅ Excellente utilisation du codebook!")
    
    # STATISTIQUES FINALES WAE-MMD
    if is_wae_mmd and wae_stats['mmd_losses']:
        avg_mmd_loss = np.mean(wae_stats['mmd_losses'])
        avg_recon_loss = np.mean(wae_stats['reconstruction_losses']) if wae_stats['reconstruction_losses'] else 0
        
        print(f"\n📊 Statistiques {model_type}:")
        print(f"   MMD Loss moyenne: {avg_mmd_loss:.6f}")
        if avg_recon_loss > 0:
            print(f"   Reconstruction Loss moyenne: {avg_recon_loss:.6f}")
            print(f"   Ratio MMD/Recon: {avg_mmd_loss/avg_recon_loss:.3f}")
        
        # Analyse de la qualité MMD
        if avg_mmd_loss < 0.01:
            print(f"✅ Excellente correspondance des distributions (MMD très faible)")
        elif avg_mmd_loss < 0.1:
            print(f"✅ Bonne correspondance des distributions")
        else:
            print(f"💡 MMD élevée - considérer ajuster le coefficient de régularisation")
    
    # SAUVEGARDE PDF ET PNG
    if all_figures:
        from matplotlib.backends.backend_pdf import PdfPages
        
        pdf_filename = f"{save_folder}/reconstructions_{model_type.lower().replace('-', '_')}.pdf"
        with PdfPages(pdf_filename) as pdf:
            for fig, var, sample_idx in all_figures:
                pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)  # Fermer pour libérer la mémoire
        
        print(f"\n📁 Toutes les reconstructions sauvegardées dans: {pdf_filename}")
        
        # Vue d'ensemble PNG
        png_filename = f"{save_folder}/reconstructions_{model_type.lower().replace('-', '_')}.png"
        
        num_vars = len(variable_names)
        num_samples = min(5, len(test_maps))
        
        fig_combined = plt.figure(figsize=(24, 8 * num_samples))
        
        plot_idx = 1
        for sample_i in range(num_samples):
            for var_j, var in enumerate(variable_names):
                plt.subplot(num_samples, num_vars, plot_idx)
                plt.text(0.5, 0.5, f"Reconstruction {var}\nÉchantillon {sample_i+1}\n({model_type})", 
                        ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
                plt.axis('off')
                plot_idx += 1
        
        plt.suptitle(f"Vue d'ensemble des reconstructions {model_type}", fontsize=16)
        plt.tight_layout()
        plt.savefig(png_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_combined)
        
        print(f"📁 Vue d'ensemble sauvegardée dans: {png_filename}")

    def determine_video_creation_logic(avi_data, variable_names, min_frames_for_video=5):
        """
        Détermine si les vidéos doivent être créées basé sur les données disponibles
    
        Args:
            avi_data: Dictionnaire contenant les données pour chaque variable
            variable_names: Liste des noms de variables
            min_frames_for_video: Nombre minimum de frames pour créer une vidéo
    
        Returns:
            bool: True si les vidéos doivent être créées
            dict: Statistiques sur les données disponibles
        """
        stats = {
            'total_variables': len(variable_names),
            'variables_with_data': 0,
            'max_frames': 0,
            'min_frames': float('inf'),
            'avg_frames': 0,
            'frame_counts': {}
        }
    
        total_frames = 0
        valid_variables = 0
    
        for var_name in variable_names:
            if var_name in avi_data and avi_data[var_name]['originals']:
                n_frames = len(avi_data[var_name]['originals'])
                stats['frame_counts'][var_name] = n_frames
                stats['variables_with_data'] += 1
                stats['max_frames'] = max(stats['max_frames'], n_frames)
                stats['min_frames'] = min(stats['min_frames'], n_frames)
                total_frames += n_frames
                valid_variables += 1
    
        if valid_variables > 0:
            stats['avg_frames'] = total_frames / valid_variables
            stats['min_frames'] = stats['min_frames'] if stats['min_frames'] != float('inf') else 0
        else:
            stats['min_frames'] = 0
    
        # NOUVELLE LOGIQUE: Créer des vidéos si au moins une variable a assez de frames
        should_create_videos = stats['max_frames'] >= min_frames_for_video
    
        print(f"🎬 DIAGNOSTIC CRÉATION VIDÉOS:")
        print(f"   - Variables avec données: {stats['variables_with_data']}/{stats['total_variables']}")
        print(f"   - Frames par variable: {stats['frame_counts']}")
        print(f"   - Frames max: {stats['max_frames']}")
        print(f"   - Frames min: {stats['min_frames']}")
        print(f"   - Frames moyenne: {stats['avg_frames']:.1f}")
        print(f"   - Seuil minimum: {min_frames_for_video}")
        print(f"   - Création vidéos: {'✅ OUI' if should_create_videos else '❌ NON'}")
    
        return should_create_videos, stats

    def apply_video_creation_fix(avi_data, variable_names, timesteps_total, batch_size):
        """
        Applique la correction pour la création de vidéos avec système batch
    
        Args:
            avi_data: Données collectées pour les vidéos
            variable_names: Liste des variables
            timesteps_total: Nombre total de timesteps
            batch_size: Taille des batches
    
        Returns:
            bool: True si les vidéos doivent être créées
        """
    
        # Calculer le nombre théorique de batches
        expected_batches = (timesteps_total + batch_size - 1) // batch_size  # Division avec arrondi au supérieur
    
        print(f"\n🔍 ANALYSE SYSTÈME BATCH:")
        print(f"   - Timesteps total: {timesteps_total}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Batches attendus: {expected_batches}")
    
        # Vérifier les données collectées
        create_avis, stats = determine_video_creation_logic(avi_data, variable_names, min_frames_for_video=3)
    
        # Si on a moins de frames que de batches attendus, c'est probablement dû à un problème de collecte
        if stats['max_frames'] < expected_batches and stats['max_frames'] > 0:
            print(f"   ⚠️ ATTENTION: Moins de frames ({stats['max_frames']}) que de batches attendus ({expected_batches})")
            print(f"   ⚠️ Cela peut indiquer un problème dans la logique de collecte des données")
        
            # Même avec peu de frames, on peut créer une vidéo si on a au moins 2 frames
            if stats['max_frames'] >= 2:
                print(f"   🔄 CORRECTION: Activation des vidéos malgré le faible nombre de frames")
                create_avis = True
    
        # Pour les petits datasets (< 5 batches), 
        # réduire le seuil minimum
        if expected_batches <= 5 and stats['max_frames'] >= 2:
            print(f"   🔄 PETITS DATASETS: Seuil réduit pour {expected_batches} batches")
            create_avis = True
    
        return create_avis

    def integrate_video_fix_in_main_function(avi_data, variable_names, timesteps_total=200, batch_size=16):
        """
        Exemple d'intégration de la correction dans votre fonction principale
        """
        create_avis = apply_video_creation_fix(avi_data, variable_names, timesteps_total, batch_size)
    
        print(f"\n🎬 DÉCISION FINALE: {'Création des vidéos activée' if create_avis else 'Pas de vidéos créées'}")
    
        return create_avis

    # Fonction pour débugger la collecte de données
    def debug_avi_data_collection(avi_data, variable_names):
        """
        Diagnostique les données collectées pour identifier les problèmes
        """
        print(f"\n🔍 DIAGNOSTIC COLLECTE DONNÉES AVI:")
    
        for var_name in variable_names:
            if var_name in avi_data:
                data = avi_data[var_name]
                print(f"\n   Variable: {var_name}")
                print(f"   - Originals: {len(data.get('originals', []))} éléments")
                print(f"   - Reconstructions: {len(data.get('reconstructions', []))} éléments")  
                print(f"   - Errors: {len(data.get('errors', []))} éléments")
            
                if 'codes' in data:
                    print(f"   - Codes: {len(data.get('codes', []))} éléments")
            
                # Vérifier la cohérence
                lengths = [len(data.get(key, [])) for key in ['originals', 'reconstructions', 'errors']]
                if len(set(lengths)) > 1:
                    print(f"   ⚠️ INCOHÉRENCE: Longueurs différentes {lengths}")
                else:
                    print(f"   ✅ Cohérent: {lengths[0]} frames pour tous les types")
                
                # Vérifier le contenu du premier élément
                if data.get('originals'):
                    first_orig = data['originals'][0]
                    if hasattr(first_orig, 'shape'):
                        print(f"   - Shape des données: {first_orig.shape}")
                    print(f"   - Type des données: {type(first_orig)}")
            else:
                print(f"\n   Variable: {var_name} - ❌ MANQUANTE dans avi_data")

    # 1. Diagnostic des données collectées
    debug_avi_data_collection(avi_data, variable_names)

    # 2. Décision intelligente de création
    create_avis = apply_video_creation_fix(avi_data, variable_names, timesteps_total=200, batch_size=16)
        # CRÉATION DES VIDÉOS AVI - VERSION CORRIGÉE POUR RATIO D'ASPECT
    if create_avis:
        (f"\n🎬 Création des vidéos AVI avec {len(avi_data[variable_names[0]]['originals'])} frames à 2 FPS...")

        def create_video_for_variable_avi(var_name, data_dict, save_folder, is_vqvae, ds, time_indices):
            """Crée deux vidéos AVI séparées pour une variable donnée : comparaison et erreur"""
            # Limitation à 16 frames maximum
            max_frames = 16
            n_frames_original = len(originals)
            n_frames = min(max_frames, n_frames_original)

            # Limiter toutes les listes aux 16 premières frames
            originals = originals[:n_frames]
            reconstructions = reconstructions[:n_frames]
            errors = errors[:n_frames]
            if codes:
                codes = codes[:n_frames]

            if not originals:
                return None, None, "no_data"

            # Vérifier le nombre de frames
            n_frames = len(originals)
            n_time_indices = len(time_indices)
            print(f"  🔍 DIAGNOSTIC {var_name}:")
            print(f"    - Nombre d'images originales disponibles: {n_frames_original}")
            print(f"    - Nombre d'images utilisées pour vidéo: {n_frames} (max: {max_frames})")
            print(f"    - Nombre d'indices temps: {n_time_indices}")
            print(f"    - Nombre de reconstructions: {len(reconstructions)}")
            print(f"    - Nombre d'erreurs: {len(errors)}")

            # Vérification de cohérence
            if n_frames != len(reconstructions) or n_frames != len(errors):
                print(f"  ⚠️ ERREUR: Nombre de frames incohérent!")
                return None, None, "inconsistent_data"

            if n_frames == 1:
                print(f"  ⚠️ ATTENTION: Une seule frame détectée - vérifiez vos données!")

            print(f"  🎬 Création de 2 vidéos AVI pour {var_name} avec {n_frames} frames...")

            # Calculer les limites globales pour une échelle cohérente
            all_orig = np.concatenate([img[~np.isnan(img)] for img in originals])
            all_recon = np.concatenate([img[~np.isnan(img)] for img in reconstructions])
            all_errors = np.concatenate([img[~np.isnan(img)] for img in errors])

            vmin_data = min(np.min(all_orig), np.min(all_recon))
            vmax_data = max(np.max(all_orig), np.max(all_recon))
            vmax_error = np.max(all_errors)

            # Paramètres visuels
            height, width = originals[0].shape
            print(f"  📏 Dimensions des cartes: {height}H x {width}W")

            # Calculer le ratio d'aspect des données géographiques
            data_aspect_ratio = width / height
            print(f"  📐 Ratio d'aspect des données: {data_aspect_ratio:.2f}")

            # Colormaps
            cmap = plt.cm.viridis.copy()
            cmap.set_bad('lightgray')
            err_cmap = plt.cm.bwr.copy()
            err_cmap.set_bad('lightgray')

            # Coordonnées pour les axes
            lat_coords = np.linspace(-90, 90, height)
            lon_coords = np.linspace(-180, 180, width)

            lat_tick_step = max(1, height // 5)
            lon_tick_step = max(1, width // 5)

            lat_ticks = np.arange(0, height, lat_tick_step)
            lon_ticks = np.arange(0, width, lon_tick_step)
        
            lat_labels = [f"{lat_coords[min(i, len(lat_coords)-1)]:.1f}°N" for i in lat_ticks]
            lon_labels = [f"{lon_coords[min(i, len(lon_coords)-1)]:.1f}°E" for i in lon_ticks]

            # Normalisations continues
            if vmax_data > vmin_data:
                data_norm = plt.Normalize(vmin=vmin_data, vmax=vmax_data)
            else:
                data_norm = plt.Normalize(vmin=vmin_data - 0.1, vmax=vmin_data + 0.1)

            if vmax_error > 0:
                err_norm = plt.Normalize(vmin=0, vmax=vmax_error)
            else:
                err_norm = plt.Normalize(vmin=0, vmax=0.1)

            # Fonction pour configurer un axe
            def setup_axis(ax, title, xlabel="Longitude", ylabel="Latitude"):
                ax.set_title(title, fontsize=14, pad=10)
                ax.set_xlabel(xlabel, fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_xticks(lon_ticks)
                ax.set_xticklabels(lon_labels, rotation=45, fontsize=10)
                ax.set_yticks(lat_ticks)
                ax.set_yticklabels(lat_labels, fontsize=10)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.set_aspect('equal')
                ax.set_xlim(0, width)
                ax.set_ylim(0, height)

            # Configuration des dimensions pour format paysage
            subplot_height = 6
            subplot_width = subplot_height * data_aspect_ratio

            # VIDÉO 1: COMPARAISON ORIGINAL/RECONSTRUCTION
            try:
                model_suffix = "vqvae" if is_vqvae else "vae"
                comparison_path = f"{save_folder}/{var_name}_comparison_{model_suffix}.avi"
        
                # Dimensions pour la comparaison (côte à côte ou avec codes VQ)
                if is_vqvae and codes:
                    # 3 panneaux : Original, Reconstruction, Codes VQ
                    fig_width = subplot_width * 3 + 3
                    fig_height = subplot_height + 1.5
                else:
                    # 2 panneaux : Original, Reconstruction
                    fig_width = subplot_width * 2 + 2
                    fig_height = subplot_height + 1.5

                dpi = 100
                comp_video_width = int(fig_width * dpi)
                comp_video_height = int(fig_height * dpi)
                print(f"  🎥 Taille vidéo comparaison: {comp_video_width}x{comp_video_height} pixels")

                # Configuration OpenCV pour vidéo comparaison
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = 2.0
                comp_video_writer = cv2.VideoWriter(comparison_path, fourcc, fps, (comp_video_width, comp_video_height))

                if not comp_video_writer.isOpened():
                    raise RuntimeError("Impossible d'ouvrir le VideoWriter pour la comparaison")
        
                # Générer les frames de comparaison
                for frame_idx in range(n_frames):
                    # Récupérer la date correspondante
                    if frame_idx < len(time_indices):
                        current_time_idx = time_indices[frame_idx]
                    else:
                        current_time_idx = time_indices[-1] if time_indices else 0
                        print(f"    ⚠️ Pas d'indice temporel pour frame {frame_idx}, utilisation du dernier disponible")

                    # Convertir la date en format lisible
                    try:
                        if current_time_idx < len(ds.time):
                            date_value = ds.time[current_time_idx].values
                            if hasattr(date_value, 'astype'):
                                date_dt = pd.to_datetime(date_value)
                                date_str = date_dt.strftime("%Y-%m-%d")
                            else:
                                date_str = str(date_value)[:10]
                        else:
                            date_str = f"Frame {frame_idx+1}"
                    except Exception as e:
                        print(f"    ⚠️ Erreur récupération date frame {frame_idx}: {e}")
                        date_str = f"Frame {frame_idx+1}"

                    # Créer la figure pour la comparaison
                    fig = plt.figure(figsize=(fig_width, fig_height))
                    fig.patch.set_facecolor('white')

                    # Titre général
                    model_type_str = "VQ-VAE" if is_vqvae else "VAE"
                    fig.suptitle(f"{var_name} - Comparaison {model_type_str} - {date_str} (Frame {frame_idx+1}/{n_frames})", 
                        fontsize=16, fontweight='bold', y=0.95)

                    # Layout des axes
                    if is_vqvae and codes and frame_idx < len(codes):
                        ax1 = plt.subplot(1, 3, 1)
                        ax2 = plt.subplot(1, 3, 2)
                        ax3 = plt.subplot(1, 3, 3)
                    else:
                        ax1 = plt.subplot(1, 2, 1)
                        ax2 = plt.subplot(1, 2, 2)
                        ax3 = None

                    # Panel 1: Original
                    im1 = ax1.imshow(originals[frame_idx], cmap=cmap, norm=data_norm,
                            aspect='auto', origin='lower', 
                            extent=[0, width, 0, height])
                    setup_axis(ax1, 'Original')

                    # Colorbar pour original
                    divider1 = make_axes_locatable(ax1)
                    cax1 = divider1.append_axes("right", size="3%", pad=0.1)
                    cbar1 = plt.colorbar(im1, cax=cax1)
                    cbar1.ax.tick_params(labelsize=10)

                    # Panel 2: Reconstruction
                    im2 = ax2.imshow(reconstructions[frame_idx], cmap=cmap, norm=data_norm,
                            aspect='auto', origin='lower', 
                            extent=[0, width, 0, height])
                    setup_axis(ax2, f'Reconstruit ({model_type_str})')

                    # Colorbar pour reconstruction
                    divider2 = make_axes_locatable(ax2)
                    cax2 = divider2.append_axes("right", size="3%", pad=0.1)
                    cbar2 = plt.colorbar(im2, cax=cax2)
                    cbar2.ax.tick_params(labelsize=10)

                    # Panel 3: Codes VQ (si applicable)
                    if ax3 is not None:
                        im3 = ax3.imshow(codes[frame_idx], cmap='tab20',
                               aspect='auto', origin='lower', 
                               extent=[0, width, 0, height])
                        setup_axis(ax3, 'Codes VQ utilisés')

                        # Colorbar pour codes
                        divider3 = make_axes_locatable(ax3)
                        cax3 = divider3.append_axes("right", size="3%", pad=0.1)
                        cbar3 = plt.colorbar(im3, cax=cax3, label='Index du code')
                        cbar3.ax.tick_params(labelsize=10)

                    # Ajustement de l'espacement
                    plt.tight_layout(pad=1.0, rect=[0, 0, 1, 0.92])
                    plt.subplots_adjust(wspace=0.3, top=0.88)

                    # Convertir en frame OpenCV
                    fig.canvas.draw()
                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    if buf.shape[:2] != (comp_video_height, comp_video_width):
                        buf = cv2.resize(buf, (comp_video_width, comp_video_height))

                    frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
                    comp_video_writer.write(frame_bgr)

                    plt.close(fig)

                    if frame_idx % max(1, n_frames // 5) == 0 or frame_idx == n_frames - 1:
                        print(f"    Comparaison - Frame {frame_idx+1}/{n_frames} écrite - {date_str}")

                comp_video_writer.release()
                print(f"  ✅ Vidéo comparaison créée: {comparison_path}")

            except Exception as e:
                print(f"  ❌ Erreur création vidéo comparaison: {e}")
                comparison_path = None

            # VIDÉO 2: ERREUR SEULE
            try:
                error_path = f"{save_folder}/{var_name}_error_{model_suffix}.avi"
        
                # Dimensions pour l'erreur (plein écran)
                fig_width = subplot_width + 1.5
                fig_height = subplot_height + 1.5

                error_video_width = int(fig_width * dpi)
                error_video_height = int(fig_height * dpi)
                print(f"  🎥 Taille vidéo erreur: {error_video_width}x{error_video_height} pixels")

                # Configuration OpenCV pour vidéo erreur
                error_video_writer = cv2.VideoWriter(error_path, fourcc, fps, (error_video_width, error_video_height))

                if not error_video_writer.isOpened():
                    raise RuntimeError("Impossible d'ouvrir le VideoWriter pour l'erreur")

                # Générer les frames d'erreur
                for frame_idx in range(n_frames):
                    # Récupérer la date (même logique que pour la comparaison)
                    if frame_idx < len(time_indices):
                        current_time_idx = time_indices[frame_idx]
                    else:
                        current_time_idx = time_indices[-1] if time_indices else 0

                    try:
                        if current_time_idx < len(ds.time):
                            date_value = ds.time[current_time_idx].values
                            if hasattr(date_value, 'astype'):
                                date_dt = pd.to_datetime(date_value)
                                date_str = date_dt.strftime("%Y-%m-%d")
                            else:
                                date_str = str(date_value)[:10]
                        else:
                            date_str = f"Frame {frame_idx+1}"
                    except:
                        date_str = f"Frame {frame_idx+1}"

                    # Créer la figure pour l'erreur
                    fig = plt.figure(figsize=(fig_width, fig_height))
                    fig.patch.set_facecolor('white')

                    # Titre général
                    fig.suptitle(f"{var_name} - Erreur absolue {model_type_str} - {date_str} (Frame {frame_idx+1}/{n_frames})", 
                        fontsize=16, fontweight='bold', y=0.95)

                    # Un seul axe pour l'erreur
                    ax = plt.subplot(1, 1, 1)

                    # Afficher l'erreur
                    im = ax.imshow(errors[frame_idx], cmap=err_cmap, norm=err_norm,
                          aspect='auto', origin='lower', 
                          extent=[0, width, 0, height])
                    setup_axis(ax, 'Erreur absolue (Original - Reconstruction)')

                    # Colorbar pour erreur
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="3%", pad=0.1)
                    cbar = plt.colorbar(im, cax=cax)
                    cbar.ax.tick_params(labelsize=10)

                    # Ajustement de l'espacement
                    plt.tight_layout(pad=1.0, rect=[0, 0, 1, 0.92])
                    plt.subplots_adjust(top=0.88)

                    # Convertir en frame OpenCV
                    fig.canvas.draw()
                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    if buf.shape[:2] != (error_video_height, error_video_width):
                        buf = cv2.resize(buf, (error_video_width, error_video_height))

                    frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
                    error_video_writer.write(frame_bgr)

                    plt.close(fig)

                    if frame_idx % max(1, n_frames // 5) == 0 or frame_idx == n_frames - 1:
                        print(f"    Erreur - Frame {frame_idx+1}/{n_frames} écrite - {date_str}")

                error_video_writer.release()
                print(f"  ✅ Vidéo erreur créée: {error_path}")

            except Exception as e:
                print(f"  ❌ Erreur création vidéo erreur: {e}")
                error_path = None

            # Fallback vers GIF si nécessaire
            if not comparison_path and not error_path:
                print(f"  🔄 Fallback vers GIF...")
                # Ici vous pouvez ajouter le code de fallback GIF si nécessaire
                return None, None, "failed"

            return comparison_path, error_path, "avi"

        # Modification de la boucle principale
        for var_name in variable_names:
            if var_name in avi_data:
                print(f"\n🎥 Traitement de la variable: {var_name}")
        
                # Déterminer le type de modèle
                is_vqvae = 'vqvae' in model_type.lower()
        
                # Appeler la fonction de création vidéo modifiée
                comparison_path, error_path, video_type = create_video_for_variable_avi(
                    var_name, 
                    avi_data[var_name], 
                    save_folder, 
                    is_vqvae, 
                    ds, 
                    time_indices
                )
            
                if comparison_path:
                    print(f"  ✅ Vidéo comparaison créée: {comparison_path}")
                if error_path:
                    print(f"  ✅ Vidéo erreur créée: {error_path}")
                if not comparison_path and not error_path:
                    print(f"  ❌ Échec création vidéos pour {var_name}")
            else:
                print(f"  ⚠️ Pas de données pour {var_name}")

        print(f"\n🎬 Création des vidéos terminée!")
    
    print(f"\n✅ Visualisation terminée. {num_samples_to_process} échantillons traités.")
    print(f"📁 Dossier de sauvegarde: {save_folder}")
    print(f"   - reconstructions_{model_type.lower().replace('-', '_')}.pdf (détaillées)")
    print(f"   - reconstructions_{model_type.lower().replace('-', '_')}.png (vue d'ensemble)")
    if create_avis:
        print(f"   - [variable]_comparison_{model_type.lower().replace('-', '_')}.avi/.gif (animations comparatives)")
    
    return model_type