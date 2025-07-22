import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from loss.vae_loss import *
from loss.vqvae_loss import *

#Entrainement VAE/WAE :
def train_vae(model, train_maps, val_maps, epochs, lr, beta, device, 
              patience, early_stopping, scheduler_gamma, 
              progressive_beta, initial_beta):
    """Entraînement VAE utilisant les masques"""
    
    model = model.to(device)
    
    # Créer l'optimiseur et le scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    
    # Suivi des pertes
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0
    best_model = None

    train_total_losses = []
    train_recon_losses = []
    train_kl_losses = []
    val_total_losses = []
    val_recon_losses = []
    val_kl_losses = []
    
    start_time = time.time()
    
    print(f"Démarrage de l'entraînement sur {len(train_maps)} cartes, {epochs} epochs...")
    print(f"Beta initial: {initial_beta}, Beta final: {beta}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        
        # Beta progressif plus doux
        current_beta = initial_beta
        if progressive_beta and epochs > 1:
            progress = epoch / (epochs - 1)  # 0 à 1
            current_beta = initial_beta + (beta - initial_beta) * progress**2  # Progression quadratique
        
        # Créer un objet de perte VAE avec le beta courant
        criterion = OceanVAELoss(kld_weight=current_beta)
        
        pbar = tqdm(train_maps, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, mask) in enumerate(pbar):
            # Préparation des données
            if len(data.shape) == 3:
                data = data.unsqueeze(0)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
                
            data = data.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass 
            recon_batch, mu, logvar = model(data, mask)
            
            # Calcul de la perte
            loss, recon, kl = criterion(recon_batch, data, mu, logvar, mask)
            
            # Vérifier la perte avant backprop
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss problématique: {loss:.6f} au batch {batch_idx}, skip")
                continue
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Suivi des pertes
            train_loss += loss.item()
            train_recon_loss += recon.item()
            train_kl_loss += kl.item()
            
            # Mise à jour de la barre de progression
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon.item():.4f}",
                'kl': f"{kl.item():.4f}",
                'beta': f"{current_beta:.4f}",
                'ocean%': f"{mask.float().mean().item()*100:.1f}"
            })
        
        # Calculer les moyennes
        num_batches = len(train_maps)
        train_loss = train_loss / num_batches
        train_recon_loss = train_recon_loss / num_batches
        train_kl_loss = train_kl_loss / num_batches

        train_total_losses.append(train_loss)
        train_recon_losses.append(train_recon_loss)
        train_kl_losses.append(train_kl_loss)

        # VALIDATION
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        
        with torch.no_grad():
            for data, mask in val_maps:
                # Même préparation que pour l'entraînement
                if len(data.shape) == 3:
                    data = data.unsqueeze(0)
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0)
                    
                data = data.to(device)
                mask = mask.to(device)
                
                recon_batch, mu, logvar = model(data, mask)
                loss, recon, kl = criterion(recon_batch, data, mu, logvar, mask)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_recon_loss += recon.item()
                    val_kl_loss += kl.item()
            
            val_loss = val_loss / len(val_maps)
            val_recon_loss = val_recon_loss / len(val_maps)
            val_kl_loss = val_kl_loss / len(val_maps)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_total_losses.append(val_loss)
        val_recon_losses.append(val_recon_loss)
        val_kl_losses.append(val_kl_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train loss: {train_loss:.6f} (Recon: {train_recon_loss:.6f}, KL: {train_kl_loss:.6f}), "
              f"Val loss: {val_loss:.6f}, "
              f"Beta: {current_beta:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model = copy.deepcopy(model.state_dict())
            print(f"Nouveau meilleur modèle (val_loss: {val_loss:.6f})")
        else:
            counter += 1
            
        if early_stopping and counter >= patience:
            print(f"Early stopping après {epoch+1} epochs")
            break
    
    # Charger le meilleur modèle
    if best_model is not None:
        model.load_state_dict(best_model)
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Temps total d'entraînement: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    return (model, 
            train_total_losses, train_recon_losses, train_kl_losses,
            val_total_losses, val_recon_losses, val_kl_losses, 
            best_val_loss)
    
#Entrainement VQVAE
def train_vqvae(model, train_maps, val_maps, epochs, lr, beta, device, 
              patience, early_stopping, scheduler_gamma, 
              progressive_beta, initial_beta):
    """
    Fonction d'entraînement adaptée pour VQ-VAE
    
    Args:
        model: Instance d'OceanVQVAE 
        train_maps: Dataset d'entraînement [(data, mask), ...]
        val_maps: Dataset de validation [(data, mask), ...]
        epochs: Nombre d'époques
        lr: Learning rate
        beta: Poids de la KL divergence (VAE) ou commitment loss (VQ-VAE)
        device: Device (cuda/cpu)
        patience: Patience pour early stopping
        early_stopping: Bool pour activer early stopping
        scheduler_gamma: Facteur de decay du learning rate
        progressive_beta: Bool pour beta progressif (ignoré pour VQ-VAE)
        initial_beta: Beta initial (ignoré pour VQ-VAE)
    
    Returns:
        (model, train_total_losses, train_recon_losses, train_kl_or_vq_losses,
         val_total_losses, val_recon_losses, val_kl_or_vq_losses, best_val_loss)
    """
    
    model = model.to(device)
    
    # Détection automatique du type de modèle
    is_vqvae = hasattr(model, 'vq_layer') and hasattr(model, 'num_embeddings')
    
    if is_vqvae:
        print(f" Détection: Modèle VQ-VAE avec {model.num_embeddings} codes")
        print(f" Beta (commitment loss): {beta}")
    else:
        print(f" Détection: Modèle VAE classique")
        print(f" Beta progressif: {progressive_beta}, Initial: {initial_beta}, Final: {beta}")
    
    # Créer l'optimiseur et le scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    
    # Suivi des pertes 
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0
    best_model = None

    train_total_losses = []
    train_recon_losses = []
    train_kl_losses = []  # ce sera les VQ losses
    val_total_losses = []
    val_recon_losses = []
    val_kl_losses = []    # ce sera les VQ losses
    
    # Suivi spécifique VQ-VAE
    if is_vqvae:
        codebook_usage_history = []
    
    start_time = time.time()
    
    print(f" Démarrage de l'entraînement sur {len(train_maps)} cartes, {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0  # VQ loss pour VQ-VAE
        
        # Beta progressif (seulement pour VAE classique)
        current_beta = beta
        if not is_vqvae and progressive_beta and epochs > 1:
            progress = epoch / (epochs - 1)
            current_beta = initial_beta + (beta - initial_beta) * progress**2
        
        # Créer le criterion approprié selon le type de modèle
        if not is_vqvae:
            # Pour VAE classique
            # criterion = OceanVAELoss(kld_weight=current_beta)
            pass
        
        # Suivi spécifique VQ-VAE
        if is_vqvae:
            used_codes = set()
        
        pbar = tqdm(train_maps, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, mask) in enumerate(pbar):
            # Préparation des données
            if len(data.shape) == 3:
                data = data.unsqueeze(0)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
                
            data = data.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            if is_vqvae:
                # Forward pass VQ-VAE
                outputs = model(data, mask)
                recon_batch, input_batch, vq_loss, encoding_indices = outputs
                
                # Calcul de la loss VQ-VAE
                loss_dict = model.loss_function(recon_batch, input_batch, vq_loss, mask=mask)
                loss = loss_dict['loss']
                recon = loss_dict['Reconstruction_Loss']
                kl = loss_dict['VQ_Loss']  # On mappe VQ_Loss vers kl pour compatibilité
                
                # Suivi des codes utilisés
                used_codes.update(encoding_indices.cpu().flatten().numpy())
                
            else:
                # Forward pass VAE classique
                recon_batch, mu, logvar = model(data, mask)
                
                # Calcul de la perte VAE classique
                loss, recon, kl = criterion(recon_batch, data, mu, logvar, mask)
            
            # Vérifier la perte avant backprop
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss problématique: {loss:.6f} au batch {batch_idx}, skip")
                continue
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Suivi des pertes
            train_loss += loss.item()
            train_recon_loss += recon.item()
            train_kl_loss += kl.item()
            
            # Mise à jour de la barre de progression
            if is_vqvae:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'recon': f"{recon.item():.4f}",
                    'vq': f"{kl.item():.4f}",  # kl contient VQ loss
                    'codes': len(used_codes),
                    'ocean%': f"{mask.float().mean().item()*100:.1f}"
                })
            else:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'recon': f"{recon.item():.4f}",
                    'kl': f"{kl.item():.4f}",
                    'beta': f"{current_beta:.4f}",
                    'ocean%': f"{mask.float().mean().item()*100:.1f}"
                })
        
        # Calculer les moyennes
        num_batches = len(train_maps)
        train_loss = train_loss / num_batches
        train_recon_loss = train_recon_loss / num_batches
        train_kl_loss = train_kl_loss / num_batches

        train_total_losses.append(train_loss)
        train_recon_losses.append(train_recon_loss)
        train_kl_losses.append(train_kl_loss)
        
        # Enregistrer l'utilisation du codebook pour VQ-VAE
        if is_vqvae:
            codebook_utilization = len(used_codes) / model.num_embeddings
            codebook_usage_history.append(codebook_utilization)

        # VALIDATION
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        
        if is_vqvae:
            val_used_codes = set()
        
        with torch.no_grad():
            for data, mask in val_maps:
                # Même préparation que pour l'entraînement
                if len(data.shape) == 3:
                    data = data.unsqueeze(0)
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0)
                    
                data = data.to(device)
                mask = mask.to(device)
                
                if is_vqvae:
                    # Forward pass VQ-VAE
                    outputs = model(data, mask)
                    recon_batch, input_batch, vq_loss_val, encoding_indices = outputs
                    
                    # Calcul des losses
                    loss_dict = model.loss_function(recon_batch, input_batch, vq_loss_val, mask=mask)
                    loss = loss_dict['loss']
                    recon = loss_dict['Reconstruction_Loss']
                    kl = loss_dict['VQ_Loss']
                    
                    val_used_codes.update(encoding_indices.cpu().flatten().numpy())
                    
                else:
                    # Forward pass VAE classique
                    recon_batch, mu, logvar = model(data, mask)
                    loss, recon, kl = criterion(recon_batch, data, mu, logvar, mask)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_recon_loss += recon.item()
                    val_kl_loss += kl.item()
            
            val_loss = val_loss / len(val_maps)
            val_recon_loss = val_recon_loss / len(val_maps)
            val_kl_loss = val_kl_loss / len(val_maps)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_total_losses.append(val_loss)
        val_recon_losses.append(val_recon_loss)
        val_kl_losses.append(val_kl_loss)
        
        # Affichage adapté selon le type de modèle
        if is_vqvae:
            val_codebook_utilization = len(val_used_codes) / model.num_embeddings
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train loss: {train_loss:.6f} (Recon: {train_recon_loss:.6f}, VQ: {train_kl_loss:.6f}), "
                  f"Val loss: {val_loss:.6f}, "
                  f"Codebook: {codebook_utilization:.1%}, LR: {scheduler.get_last_lr()[0]:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train loss: {train_loss:.6f} (Recon: {train_recon_loss:.6f}, KL: {train_kl_loss:.6f}), "
                  f"Val loss: {val_loss:.6f}, "
                  f"Beta: {current_beta:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model = copy.deepcopy(model.state_dict())
            print(f"Nouveau meilleur modèle (val_loss: {val_loss:.6f})")
        else:
            counter += 1
            
        # Warning spécifique VQ-VAE
        if is_vqvae and codebook_utilization < 0.1:
            print(f"  Faible utilisation du codebook: {codebook_utilization:.1%}")
            
        if early_stopping and counter >= patience:
            print(f"Early stopping après {epoch+1} epochs")
            break
    
    # Charger le meilleur modèle
    if best_model is not None:
        model.load_state_dict(best_model)
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Temps total d'entraînement: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Affichage final spécifique VQ-VAE
    if is_vqvae:
        print(f" Utilisation finale du codebook: {codebook_usage_history[-1]:.1%}")
    
    return (model, 
            train_total_losses, train_recon_losses, train_kl_losses,
            val_total_losses, val_recon_losses, val_kl_losses, 
            best_val_loss)

#Entrainement WAE
def train_wae(model, train_maps, val_maps, epochs, lr, beta, device, 
              patience, early_stopping, scheduler_gamma, 
              progressive_beta, initial_beta, lambda_reg):
    """
    Fonction d'entraînement WAE-MMD compatible avec ton interface main().
    
    Args:
        model: Instance OceanWAE
        train_maps: Dataset d'entraînement [(data, mask), ...]
        val_maps: Dataset de validation [(data, mask), ...]
        epochs: Nombre d'époques
        lr: Learning rate
        beta: Paramètre ignoré (maintenu pour compatibilité)
        device: Device (cuda/cpu)
        patience: Patience pour early stopping
        early_stopping: Bool pour activer early stopping
        scheduler_gamma: Facteur de decay du learning rate
        progressive_beta: Ignoré pour WAE
        initial_beta: Ignoré pour WAE
        lambda_reg: Poids de la régularisation MMD (remplace beta)
    
    Returns:
        Tuple compatible avec ton main(): 
        (model, train_total_losses, train_recon_losses, train_mmd_losses,
         val_total_losses, val_recon_losses, val_mmd_losses, best_val_loss)
    """
    
    model = model.to(device)
    
    # Mise à jour du lambda_reg si le modèle le supporte
    if hasattr(model, 'lambda_reg'):
        model.lambda_reg = lambda_reg
    
    print(f" Entraînement WAE-MMD Océanique:")
    print(f"   - Learning rate: {lr}")
    print(f"   - Lambda regularization: {lambda_reg} (remplace beta)")
    print(f"   - Sigma kernels: {model.sigma_list if hasattr(model, 'sigma_list') else 'default'}")
    
    # Créer l'optimiseur et le scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    
    # Suivi des pertes
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0
    best_model = None

    train_total_losses = []
    train_recon_losses = []
    train_kl_losses = []  # Sera les MMD losses 
    val_total_losses = []
    val_recon_losses = []
    val_kl_losses = []    # Sera les MMD losses
    
    start_time = time.time()
    
    print(f"Démarrage de l'entraînement sur {len(train_maps)} cartes, {epochs} epochs...")
    print(f"Lambda MMD: {lambda_reg} (pas de beta progressif pour WAE)")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_mmd_loss = 0
        
        pbar = tqdm(train_maps, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, mask) in enumerate(pbar):
            # Préparation des données (identique à ton VAE)
            if len(data.shape) == 3:
                data = data.unsqueeze(0)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
                
            data = data.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass WAE - DIFFÉRENCE: retourne [reconstruction, input, z]
            outputs = model(data, mask)
            recon_batch, input_batch, z = outputs
            
            # Calcul de la perte WAE
            loss_dict = model.loss_function(recon_batch, input_batch, z, mask=mask)
            loss = loss_dict['loss']
            recon = loss_dict['reconstruction_loss']
            mmd = loss_dict['mmd_loss']
            
            # Vérifier la perte avant backprop
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss problématique: {loss:.6f} au batch {batch_idx}, skip")
                continue
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Suivi des pertes
            train_loss += loss.item()
            train_recon_loss += recon.item()
            train_mmd_loss += mmd.item()
            
            # Mise à jour de la barre de progression
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon.item():.4f}",
                'mmd': f"{mmd.item():.4f}",
                'lambda': f"{lambda_reg:.2f}",
                'ocean%': f"{mask.float().mean().item()*100:.1f}"
            })
        
        # Calculer les moyennes
        num_batches = len(train_maps)
        train_loss = train_loss / num_batches
        train_recon_loss = train_recon_loss / num_batches
        train_mmd_loss = train_mmd_loss / num_batches

        train_total_losses.append(train_loss)
        train_recon_losses.append(train_recon_loss)
        train_kl_losses.append(train_mmd_loss)  # Mapping MMD → kl pour compatibilité

        # VALIDATION
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_mmd_loss = 0
        
        with torch.no_grad():
            for data, mask in val_maps:
                # Même préparation que pour l'entraînement
                if len(data.shape) == 3:
                    data = data.unsqueeze(0)
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0)
                    
                data = data.to(device)
                mask = mask.to(device)
                
                # Forward pass validation
                outputs = model(data, mask)
                recon_batch, input_batch, z = outputs
                
                # Calcul des losses
                loss_dict = model.loss_function(recon_batch, input_batch, z, mask=mask)
                loss = loss_dict['loss']
                recon = loss_dict['reconstruction_loss']
                mmd = loss_dict['mmd_loss']
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_recon_loss += recon.item()
                    val_mmd_loss += mmd.item()
            
            val_loss = val_loss / len(val_maps)
            val_recon_loss = val_recon_loss / len(val_maps)
            val_mmd_loss = val_mmd_loss / len(val_maps)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_total_losses.append(val_loss)
        val_recon_losses.append(val_recon_loss)
        val_kl_losses.append(val_mmd_loss)  # Mapping MMD → kl pour compatibilité
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train loss: {train_loss:.6f} (Recon: {train_recon_loss:.6f}, MMD: {train_mmd_loss:.6f}), "
              f"Val loss: {val_loss:.6f}, "
              f"Lambda: {lambda_reg:.2f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Diagnostics WAE spécifiques
        if train_mmd_loss < 0.001:
            print(f"  MMD très faible ({train_mmd_loss:.6f}) - considérer augmenter lambda_reg")
        elif train_mmd_loss > 1.0:
            print(f"  MMD très élevé ({train_mmd_loss:.6f}) - considérer réduire lambda_reg")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model = copy.deepcopy(model.state_dict())
            print(f"Nouveau meilleur modèle (val_loss: {val_loss:.6f})")
        else:
            counter += 1
            
        if early_stopping and counter >= patience:
            print(f"Early stopping après {epoch+1} epochs")
            break
    
    # Charger le meilleur modèle
    if best_model is not None:
        model.load_state_dict(best_model)
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Temps total d'entraînement: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Statistiques finales WAE
    print(f"\n Statistiques finales WAE-MMD:")
    print(f"   - MMD final: {train_mmd_loss:.6f}")
    print(f"   - Lambda utilisé: {lambda_reg}")
    print(f"   - Meilleure validation loss: {best_val_loss:.6f}")
    
    # Analyse de l'espace latent WAE
    model.eval()
    with torch.no_grad():
        sample_data, sample_mask = next(iter(train_maps))
        if len(sample_data.shape) == 3:
            sample_data = sample_data.unsqueeze(0)
        
        sample_data = sample_data.to(device)
        z_sample = model.encode(sample_data, sample_mask)
        z_flat = z_sample.view(z_sample.size(0), -1)
        
        print(f"   - Espace latent - Moyenne: {z_flat.mean().item():.4f}, Std: {z_flat.std().item():.4f}")
        
        # Vérifier si l'espace latent suit une distribution gaussienne
        z_mean_target = 0.0
        z_std_target = 1.0
        latent_quality = abs(z_flat.mean().item() - z_mean_target) + abs(z_flat.std().item() - z_std_target)
        
        if latent_quality < 0.5:
            print(f"✅ Espace latent bien régularisé (qualité: {latent_quality:.3f})")
        else:
            print(f"  Espace latent mal régularisé (qualité: {latent_quality:.3f}) - ajuster lambda_reg")
    
    return (model, 
            train_total_losses, train_recon_losses, train_kl_losses,
            val_total_losses, val_recon_losses, val_kl_losses, 
            best_val_loss)