import os as os
import matplotlib.pyplot as plt
import cv2
print(f"✅ OpenCV version: {cv2.__version__}")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
import argparse
import seaborn as sns
import sys 


# ============= FONCTION UTILITAIRE POUR VALIDATION CONFIG =============
def validate_config():
    """
    Valide la configuration avant le lancement
    """
    global model_to_use, variables, epochs, learning_rate, beta, MODELS
    
    # Vérifications de base
    available_models = [m.upper() for m in MODELS if m.upper() in ["VAE", "VQVAE", "WAE-MMD"]]
    assert model_to_use.upper() in available_models, f"Modèle '{model_to_use}' non supporté. Disponibles: {available_models}"
    assert epochs > 0, "Le nombre d'époques doit être positif"
    assert learning_rate > 0, "Le learning rate doit être positif"
    assert beta >= 0, "Beta doit être positif ou nul"
    
    # Vérifications spécifiques
    if model_to_use.upper() == "VAE" or model_to_use.upper() == "WAE-MMD":
        if progressive_beta:
            assert initial_beta <= beta, "initial_beta doit être <= beta final"
    
    print(f" Configuration validée pour {model_to_use.upper()}")
    


# Cellule 1: Définir le mode d'exécution
# Choisissez l'un des modes suivants en décommentant la ligne correspondante

# Mode 1: Entraînement normal avec sauvegarde automatique
#mode = "training"

# Mode 2: Chargement d'un modèle existant pour évaluation
# mode = "load_and_evaluate"
# model_path = "results_vae_20240315_143022/ocean_vae_model.pth"

# Mode 3: Inférence rapide uniquement
# mode = "inference_only"
# model_path = "results_vae_20240315_143022/latest_model.pth"


# ============= FONCTIONS UTILITAIRES  =============

def quick_inference(model_path, test_data=None):
    """
    Fonction rapide pour charger un modèle et faire de l'inférence
    Parfait pour Jupyter Notebook
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats_dict, checkpoint = load_model_universal(model_path, device)
    
    print(f" Modèle {checkpoint['model_type']} chargé")
    print(f" Variables: {checkpoint.get('variables', [])}")
    
    if test_data is not None:
        print(" Inférence en cours...")
        model.eval()
        with torch.no_grad():
            if checkpoint['model_type'] in ["VAE", "WAE-MMD"]:
                reconstructed, mu, logvar = model(test_data.to(device))
                return model, reconstructed, mu, logvar
            elif checkpoint['model_type'] == "VQVAE":
                reconstructed, vq_loss, perplexity, encoding_indices = model(test_data.to(device))
                return model, reconstructed, vq_loss, perplexity, encoding_indices
    
    return model, stats_dict

def list_saved_models(base_dir="./"):
    """
    Liste tous les modèles sauvegardés dans le répertoire
    """
    import glob
    
    model_files = glob.glob(os.path.join(base_dir, "**/ocean_*_model.pth"), recursive=True)
    latest_files = glob.glob(os.path.join(base_dir, "**/latest_model.pth"), recursive=True)
    
    print(" Modèles sauvegardés trouvés:")
    print("\n Modèles spécifiques:")
    for i, model_file in enumerate(model_files, 1):
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            model_type = checkpoint.get('model_type', 'Unknown')
            timestamp = checkpoint.get('timestamp', 'Unknown')
            print(f"   {i}. {model_file}")
            print(f"      Type: {model_type} | Timestamp: {timestamp}")
        except:
            print(f"   {i}. {model_file} ( Erreur de lecture)")
    
    print("\n Modèles 'latest':")
    for i, latest_file in enumerate(latest_files, 1):
        try:
            checkpoint = torch.load(latest_file, map_location='cpu')
            model_type = checkpoint.get('model_type', 'Unknown')
            timestamp = checkpoint.get('timestamp', 'Unknown')
            print(f"   {i}. {latest_file}")
            print(f"      Type: {model_type} | Timestamp: {timestamp}")
        except:
            print(f"   {i}. {latest_file} ( Erreur de lecture)")

def compare_models(model_paths):
    """
    Compare plusieurs modèles sauvegardés
    """
    print(" Comparaison des modèles:")
    
    for i, model_path in enumerate(model_paths, 1):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"\n Modèle {i}: {os.path.basename(model_path)}")
            print(f"   Type: {checkpoint.get('model_type', 'Unknown')}")
            print(f"   Variables: {checkpoint.get('variables', [])}")
            print(f"   Timestamp: {checkpoint.get('timestamp', 'Unknown')}")
            
            if 'best_val_loss' in checkpoint:
                print(f"   Meilleure val loss: {checkpoint['best_val_loss']:.6f}")
            
            config = checkpoint.get('config', {})
            print(f"   Config: lr={config.get('learning_rate', 'N/A')}, "
                  f"beta={config.get('beta', 'N/A')}, "
                  f"epochs={config.get('epochs', 'N/A')}")
                  
        except Exception as e:
            print(f" Erreur lors du chargement de {model_path}: {e}")


# =============  INFÉRENCE SUR NOUVELLES DONNÉES =============

def run_inference_on_new_data(model_path, new_data_tensor, mask_tensor=None):
    """
    Fonction complète pour faire de l'inférence sur de nouvelles données
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats_dict, checkpoint = load_model_universal(model_path, device)
    
    print(f"🔍 Inférence avec modèle {checkpoint['model_type']}")
    
    # Préparation des données
    new_data_tensor = new_data_tensor.to(device)
    if mask_tensor is not None:
        mask_tensor = mask_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        if checkpoint['model_type'] in ["VAE", "WAE-MMD"]:
            if mask_tensor is not None:
                reconstructed, mu, logvar = model(new_data_tensor, mask_tensor)
            else:
                reconstructed, mu, logvar = model(new_data_tensor)
            
            print(f"✅ Reconstruction terminée")
            print(f"   Shape originale: {new_data_tensor.shape}")
            print(f"   Shape reconstruite: {reconstructed.shape}")
            print(f"   Shape latente: {mu.shape}")
            
            return reconstructed, mu, logvar
            
        elif checkpoint['model_type'] == "VQVAE":
            if mask_tensor is not None:
                reconstructed, vq_loss, perplexity, encoding_indices = model(new_data_tensor, mask_tensor)
            else:
                reconstructed, vq_loss, perplexity, encoding_indices = model(new_data_tensor)
            
            print(f"✅ Reconstruction terminée")
            print(f"   Shape originale: {new_data_tensor.shape}")
            print(f"   Shape reconstruite: {reconstructed.shape}")
            print(f"   Perplexité: {perplexity.item():.2f}")
            print(f"   Codes utilisés: {len(torch.unique(encoding_indices))}")
            
            return reconstructed, vq_loss, perplexity, encoding_indices


# ============= VISUALISATION RAPIDE =============

def quick_visualization(model_path, test_data, test_mask=None, variable_names=None):
    """
    Visualisation rapide des résultats d'inférence
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats_dict, checkpoint = load_model_universal(model_path, device)
    
    # Inférence
    model.eval()
    with torch.no_grad():
        if checkpoint['model_type'] in ["VAE", "WAE-MMD"]:
            reconstructed, mu, logvar = model(test_data.to(device))
        elif checkpoint['model_type'] == "VQVAE":
            reconstructed, vq_loss, perplexity, encoding_indices = model(test_data.to(device))
    
    # Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(test_data[0, 0].cpu().numpy(), cmap='viridis')
    axes[0, 0].set_title('Original - Variable 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(test_data[0, 1].cpu().numpy(), cmap='viridis')
    axes[0, 1].set_title('Original - Variable 2')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(test_data[0, 2].cpu().numpy(), cmap='viridis')
    axes[0, 2].set_title('Original - Variable 3')
    axes[0, 2].axis('off')
    
    # Reconstruit
    axes[1, 0].imshow(reconstructed[0, 0].cpu().numpy(), cmap='viridis')
    axes[1, 0].set_title('Reconstruit - Variable 1')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(reconstructed[0, 1].cpu().numpy(), cmap='viridis')
    axes[1, 1].set_title('Reconstruit - Variable 2')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(reconstructed[0, 2].cpu().numpy(), cmap='viridis')
    axes[1, 2].set_title('Reconstruit - Variable 3')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Résultats {checkpoint["model_type"]}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return reconstructed

print(" Toutes les fonctions utilitaires sont prêtes!")
print(" Adaptez les chemins et variables selon vos besoins")