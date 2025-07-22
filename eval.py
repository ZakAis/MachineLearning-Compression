import numpy as np
from loss.vae_loss import *
from visualisation import *

#Evaluation VAE, valable pour tout les modèles VAE, WAE, VQVAE
def evaluate_vae_on_test(model, test_maps, beta, device, variable_names, time_indices, progressive_beta):
    """
    Évalue le modèle VAE sur l'ensemble de test avec support des masques.
    
    Paramètres:
    -----------
    model : nn.Module
        Modèle VAE à évaluer
    test_maps : list
        Liste de tuples (data, mask)
    beta : float
        Coefficient de pondération pour le terme KL
    device : torch.device
        Appareil sur lequel effectuer les calculs
    variable_names : list
        Liste des noms des variables utilisées
    time_indices : list, optional
        Liste des indices temporels correspondant aux patches de test
    progressive_beta : bool, optional
        Si True, utilise une augmentation progressive du coefficient beta
        
    Retourne:
    ---------
    test_loss, test_recon_loss, test_kl_loss : float
        Pertes moyennes sur l'ensemble de test
    """
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    
    # Créer un objet de perte VAE
    criterion = OceanVAELoss(kld_weight=beta)
    
    with torch.no_grad():
        for i, (data, mask) in enumerate(test_maps):
            # Assurer que data a la bonne forme [B, C, H, W]
            if len(data.shape) == 3:  # Si [C, H, W]
                data = data.unsqueeze(0)  # Ajouter dimension batch
            if len(mask.shape) == 2:  # Si [H, W]
                mask = mask.unsqueeze(0)  # Ajouter dimension batch
            
            data = data.to(device)
            mask = mask.to(device)
                
            # Forward pass avec le masque
            recon_batch, mu, logvar = model(data, mask)
            
            # Ajuster le coefficient beta si progressif
            current_beta = min(1.0, (i+1)/len(test_maps) * beta) if progressive_beta else beta
            criterion.kld_weight = current_beta
            
            # Calculer la perte avec la classe OceanVAELoss
            total_loss, recon, kl = criterion(recon_batch, data, mu, logvar, mask)
            
            test_loss += total_loss.item()
            test_recon_loss += recon.item()
            test_kl_loss += kl.item()
    
    # Calculer les moyennes
    n_samples = len(test_maps)
    test_loss /= n_samples
    test_recon_loss /= n_samples
    test_kl_loss /= n_samples
    
    print(f"\nRésultats sur l'ensemble de test avec beta={beta}:")
    print(f"Perte totale: {test_loss:.6f}")
    print(f"Perte de reconstruction: {test_recon_loss:.6f}")
    print(f"Perte KL: {test_kl_loss:.6f}")
    
    return test_loss, test_recon_loss, test_kl_loss