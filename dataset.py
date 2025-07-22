import torch
from torch.utils.data import DataLoader, Dataset, Subset
import xarray as xr
import numpy as np
import torch
from typing import List, Optional
import psutil
import gc
import warnings
import pandas as pd  # Import nécessaire pour les dates
from collections import Counter
import random
import os
from datetime import datetime

class OceanDataLoader:
    """
    Classe pour charger et préprocesser les données océaniques GLORYS.
    
    Permet de charger des données depuis différentes sources (URLs, fichiers locaux)
    et de les préprocesser pour l'entraînement de modèles VAE/VQ-VAE.
    """
    
    def __init__(self, base_url: str = "https://minio.dive.edito.eu/project-ml-compression/public/data/baby_GLORYS"):
        """
        Initialise le OceanDataLoader.
        
        Args:
            base_url: URL de base pour les données GLORYS
        """
        self.base_url = base_url
        self.dataset = None
        self.reshaped_dataset = None
        self._lazy_datasets = {}  # Cache des datasets individuels
        self._lazy_mode = False
        self._years_loaded = []
        
        # Mapping des canaux pour reshape_dataset
        self.channel_mapping = {
            0: ('zos', None),
            1: ('thetao', 0),
            2: ('thetao', 1),
            3: ('so', 0),
            4: ('so', 1),
            5: ('uo', 0),
            6: ('uo', 1),
            7: ('vo', 0),
            8: ('vo', 1),
        }
    
    def load_years(self, years: List[int], max_retries: int = 3) -> xr.Dataset:
        """
        Charge et concatène les données pour les années spécifiées.
        Mode lazy: garde les références aux datasets individuels.
        
        Args:
            years: Liste des années à charger
            max_retries: Nombre maximum de tentatives par fichier
            
        Returns:
            Dataset xarray concaténé (ou référence lazy)
        """
        print(f" Chargement de {len(years)} années de données GLORYS ({years[0]}-{years[-1]})")
        
        # Vérifier si on peut charger tout en mémoire
        memory = psutil.virtual_memory()
        available_gb = memory.available / 1e9
        
        # Estimation plus réaliste basée sur les vraies dimensions
        # Prendre un échantillon pour connaître les vraies dimensions
        sample_url = f"{self.base_url}/{years[0]}.zarr"
        try:
            sample_ds = xr.open_zarr(sample_url, consolidated=True)
            if 'data' in sample_ds:
                actual_shape = sample_ds['data'].shape
                lat_size, lon_size = actual_shape[-2], actual_shape[-1]
            else:
                lat_size, lon_size = 336, 720  # Fallback basé sur vos données
            sample_ds.close()
        except:
            lat_size, lon_size = 336, 720  # Fallback
        
        estimated_size = len(years) * 365 * 9 * lat_size * lon_size * 4 / 1e9
        print(f" Dimensions détectées: {lat_size} x {lon_size}")
        
        if estimated_size > available_gb * 0.3:  # Plus conservateur
            print(f"  Mode LAZY activé (estimation: {estimated_size:.1f}GB > {available_gb*0.6:.1f}GB disponible)")
            self._lazy_mode = True
            return self._load_years_lazy(years, max_retries)
        else:
            print(f"✅ Mode NORMAL (estimation: {estimated_size:.1f}GB)")
            self._lazy_mode = False
            return self._load_years_normal(years, max_retries)
    
    def _load_years_lazy(self, years: List[int], max_retries: int = 3) -> xr.Dataset:
        """Version lazy loading qui ne charge pas tout en mémoire"""
        print(" Chargement LAZY des datasets...")
        
        # Stocker les URLs et métadonnées sans charger les données
        self._lazy_datasets = {}
        failed_years = []
        sample_ds = None
        
        for year in years:
            url = f"{self.base_url}/{year}.zarr"
            
            for attempt in range(max_retries):
                try:
                    print(f"  📅 {year}... ", end="")
                    # Charger seulement les métadonnées
                    ds = xr.open_zarr(url, consolidated=True)
                    
                    # Garder une référence sans charger les données
                    self._lazy_datasets[year] = {
                        'url': url,
                        'dataset': ds,
                        'loaded': False
                    }
                    
                    if sample_ds is None:
                        sample_ds = ds
                    
                    print(f"✅ (lazy)")
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  Retry...")
                    else:
                        print(f" {e}")
                        failed_years.append(year)
        
        self._years_loaded = [y for y in years if y not in failed_years]
        
        print(f" Lazy loading: {len(self._years_loaded)}/{len(years)} années référencées")
        
        if not self._lazy_datasets:
            raise ValueError(" Aucun dataset référencé")
        
        # Créer un dataset "fantôme" avec les bonnes dimensions
        if sample_ds is not None:
            # Estimer les dimensions totales
            total_time = len(self._years_loaded) * len(sample_ds.time)
            
            # Créer les coordonnées temporelles
            time_coords = []
            for year in self._years_loaded:
                ds = self._lazy_datasets[year]['dataset']
                time_coords.extend(ds.time.values)
            
            # Dataset fantôme avec métadonnées correctes
            self.dataset = LazyDataset(self._lazy_datasets, self._years_loaded, sample_ds, time_coords)
            
            print(f" Dataset lazy créé!")
            print(f"    Shape estimée: time={len(time_coords)}")
            print(f"     Variables: {list(sample_ds.data_vars.keys())}")
            
            return self.dataset
    
    def _load_years_normal(self, years: List[int], max_retries: int = 3) -> xr.Dataset:
        """Version normale qui charge tout en mémoire (code original)"""
        urls = [f"{self.base_url}/{year}.zarr" for year in years]
        
        print(" Chargement NORMAL des datasets...")
        datasets = []
        failed_years = []
        
        for i, (year, url) in enumerate(zip(years, urls)):
            success = False
            
            for attempt in range(max_retries):
                try:
                    print(f"   {year}... ", end="")
                    ds = xr.open_zarr(url, consolidated=True)
                    datasets.append(ds)
                    print(f"✅ ({dict(ds.sizes)})")
                    success = True
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  Tentative {attempt + 1}/{max_retries} échouée, retry...")
                    else:
                        print(f" Erreur après {max_retries} tentatives: {e}")
                        failed_years.append(year)
        
        print(f" Résumé: {len(datasets)}/{len(years)} années chargées")
        if failed_years:
            print(f"    Échecs: {failed_years}")
        
        if not datasets:
            raise ValueError(" Aucun dataset chargé avec succès")
        
        print(" Concaténation des datasets...")
        try:
            self.dataset = xr.concat(datasets, dim='time')
            
            print(f" Dataset final créé!")
            print(f"    Shape: {dict(self.dataset.sizes)}")
            print(f"    Période: {self.dataset.time.values[0]} à {self.dataset.time.values[-1]}")
            
            return self.dataset
            
        except Exception as e:
            raise RuntimeError(f" Erreur lors de la concaténation: {e}")
    
    def load_subset(self, years: List[int], max_timesteps: Optional[int] = None) -> xr.Dataset:
        """
        Charge un sous-ensemble des données avec limitation temporelle.
        Compatible lazy loading.
        
        Args:
            years: Années à charger
            max_timesteps: Nombre maximum de pas de temps à conserver
            
        Returns:
            Dataset réduit
        """
        # Charger (ou référencer) toutes les données
        full_dataset = self.load_years(years)
        
        # Appliquer la limitation temporelle
        if max_timesteps is not None:
            print(f"  Limitation à {max_timesteps} pas de temps...")
            
            if self._lazy_mode:
                # En mode lazy, on marque juste la limitation
                full_dataset.max_timesteps = max_timesteps
                print(f"   📏 Limitation lazy appliquée")
            else:
                # En mode normal, on fait le slice
                subset = full_dataset.isel(time=slice(0, max_timesteps))
                print(f"   📏 Nouvelle shape: {dict(subset.sizes)}")
                self.dataset = subset
                full_dataset = subset
        
        return full_dataset
    
    def get_variable_info(self) -> dict:
        """
        Retourne des informations sur les variables du dataset.
        
        Returns:
            Dictionnaire avec les informations des variables
        """
        if self.reshaped_dataset is None:
            raise ValueError("Dataset pas encore restructuré. Utilisez reshape_dataset() d'abord.")
        
        info = {}
        for var_name in self.reshaped_dataset.data_vars:
            var = self.reshaped_dataset[var_name]
            info[var_name] = {
                'shape': var.shape,
                'dims': var.dims,
                'dtype': str(var.dtype),
                'has_nan': bool(np.isnan(var.values).any()),
                'min': float(np.nanmin(var.values)),
                'max': float(np.nanmax(var.values)),
                'mean': float(np.nanmean(var.values)),
                'std': float(np.nanstd(var.values))
            }
        
        return info
    
    def save_dataset(self, filepath: str, format: str = 'zarr') -> None:
        """
        Sauvegarde le dataset restructuré.
        
        Args:
            filepath: Chemin de sauvegarde
            format: Format de sauvegarde ('zarr' ou 'netcdf')
        """
        if self.reshaped_dataset is None:
            raise ValueError("Aucun dataset restructuré à sauvegarder")
        
        print(f" Sauvegarde du dataset en format {format}...")
        
        if format.lower() == 'zarr':
            self.reshaped_dataset.to_zarr(filepath, mode='w')
        elif format.lower() == 'netcdf':
            self.reshaped_dataset.to_netcdf(filepath)
        else:
            raise ValueError(f"Format '{format}' non supporté. Utilisez 'zarr' ou 'netcdf'.")
        
        print(f" Dataset sauvegardé: {filepath}")
    
    @classmethod
    def load_from_local(cls, filepath: str, max_timesteps: Optional[int] = None) -> 'OceanDataLoader':
        """
        Charge un dataset depuis un fichier local.
        
        Args:
            filepath: Chemin vers le fichier
            max_timesteps: Limitation temporelle optionnelle
            
        Returns:
            Instance de OceanDataLoader avec le dataset chargé
        """
        print(f" Chargement depuis fichier local: {filepath}")
        
        loader = cls()
        
        try:
            if filepath.endswith('.zarr'):
                ds = xr.open_zarr(filepath)
            elif filepath.endswith('.nc'):
                ds = xr.open_dataset(filepath)
            else:
                try:
                    ds = xr.open_zarr(filepath)
                except:
                    ds = xr.open_dataset(filepath)
            
            if max_timesteps is not None:
                ds = ds.isel(time=slice(0, max_timesteps))
                print(f"  Limité à {max_timesteps} pas de temps")
            
            loader.dataset = ds
            print(f" Dataset local chargé: {dict(ds.sizes)}")
            
            return loader
            
        except Exception as e:
            raise RuntimeError(f" Erreur lors du chargement: {e}")


class LazyDataset:
    """Dataset lazy qui charge les données à la demande"""
    
    def __init__(self, lazy_datasets, years, sample_ds, time_coords):
        self._lazy_datasets = lazy_datasets
        self._years = years
        self._sample_ds = sample_ds
        self._time_coords = time_coords
        self.max_timesteps = None
        
        # Interface compatible avec xr.Dataset
        self.dims = sample_ds.dims
        self.data_vars = sample_ds.data_vars
        self.coords = sample_ds.coords
    
    def __getitem__(self, key):
        """Accès aux variables avec chargement à la demande"""
        if key == 'data':
            return LazyDataArray(self._lazy_datasets, self._years, self._time_coords, 
                                self._sample_ds, self.max_timesteps)
        return self._sample_ds[key]
    
    @property
    def time(self):
        """Coordonnées temporelles"""
        coords = self._time_coords
        if self.max_timesteps:
            coords = coords[:self.max_timesteps]
        return xr.DataArray(coords, dims='time')


class LazyDataArray:
    """DataArray lazy pour la variable 'data'"""
    
    def __init__(self, lazy_datasets, years, time_coords, sample_ds, max_timesteps):
        self._lazy_datasets = lazy_datasets
        self._years = years
        self._time_coords = time_coords
        self._sample_ds = sample_ds
        self._max_timesteps = max_timesteps
        
        # Calculer les dimensions
        total_time = len(time_coords)
        if max_timesteps:
            total_time = min(total_time, max_timesteps)
        
        sample_data = sample_ds['data']
        self.shape = (total_time, *sample_data.shape[1:])
        self.dims = sample_data.dims
    
    @property
    def values(self):
        """NOUVEAU: Retourne un générateur au lieu de charger tout en mémoire"""
        print(" Mode GÉNÉRATEUR LAZY activé - pas de chargement complet!")
        # Retourner un objet qui se comporte comme un array mais génère les données à la demande
        return LazyDataGenerator(self._lazy_datasets, self._years, self._max_timesteps)


class LazyDataGenerator:
    """Générateur qui simule un array numpy mais génère les données à la demande"""
    
    def __init__(self, lazy_datasets, years, max_timesteps):
        self._lazy_datasets = lazy_datasets
        self._years = years
        self._max_timesteps = max_timesteps
        
        # Calculer la shape totale
        total_time = 0
        sample_shape = None
        
        for year in years:
            ds = lazy_datasets[year]['dataset']
            year_length = len(ds.time)
            total_time += year_length
            if sample_shape is None:
                sample_shape = ds['data'].shape[1:]  # Sans la dimension temps
        
        if max_timesteps:
            total_time = min(total_time, max_timesteps)
        
        self.shape = (total_time, *sample_shape)
        self.dtype = np.float32
    
    def __getitem__(self, key):
        """Permet l'accès par slice comme un vrai array numpy"""
        if isinstance(key, slice):
            return self._get_slice(key)
        else:
            return self._get_single(key)
    
    def _get_slice(self, slice_obj):
        """Récupère une tranche de données"""
        start, stop, step = slice_obj.indices(self.shape[0])
        
        if step != 1:
            raise NotImplementedError("Step != 1 pas supporté")
        
        # Charger seulement la tranche demandée
        return self._load_time_range(start, stop)
    
    def _load_time_range(self, start_time, end_time):
        """Charge une plage temporelle spécifique"""
        print(f"    Chargement tranche temporelle {start_time}-{end_time}")
        
        result_data = []
        current_time = 0
        
        for year in self._years:
            ds = self._lazy_datasets[year]['dataset']
            year_data = ds['data']
            year_length = len(ds.time)
            
            # Vérifier si cette année contient des données dans notre plage
            year_start = current_time
            year_end = current_time + year_length
            
            if year_end <= start_time:
                # Cette année est avant notre plage
                current_time = year_end
                continue
            
            if year_start >= end_time:
                # Cette année est après notre plage
                break
            
            # Calculer les indices dans cette année
            local_start = max(0, start_time - year_start)
            local_end = min(year_length, end_time - year_start)
            
            if local_start < local_end:
                print(f"     📅 Année {year}: indices {local_start}-{local_end}")
                year_slice = year_data.isel(time=slice(local_start, local_end)).values
                result_data.append(year_slice)
                
                # Nettoyage immédiat
                del year_slice
                gc.collect()
            
            current_time = year_end
            
            # Arrêt si on a atteint la limite
            if self._max_timesteps and current_time >= self._max_timesteps:
                break
        
        if result_data:
            combined = np.concatenate(result_data, axis=0)
            del result_data
            gc.collect()
            return combined
        else:
            # Retourner un array vide avec la bonne shape
            empty_shape = (0, *self.shape[1:])
            return np.empty(empty_shape, dtype=self.dtype)


def reshape_dataset(ds):
    """
    Restructure le dataset - VERSION ULTRA-LAZY qui traite par mini-chunks.
    """
    ch_map = {
        0: ('zos', None),
        1: ('thetao', 0),
        2: ('thetao', 1),
        3: ('so', 0),
        4: ('so', 1),
        5: ('uo', 0),
        6: ('uo', 1),
        7: ('vo', 0),
        8: ('vo', 1),
    }

    print(" Restructuration ULTRA-LAZY du dataset...")
    
    # Gestion lazy avec générateur
    if hasattr(ds, '_lazy_datasets'):
        print("   Mode GÉNÉRATEUR LAZY activé")
        data_generator = ds['data'].values  # C'est maintenant un LazyDataGenerator
        total_timesteps = data_generator.shape[0]
    else:
        print("   Mode NORMAL")
        data = ds['data'].values
        total_timesteps = data.shape[0]
    
    time = ds.time
    
    # Utiliser les coordonnées du dataset sample
    if hasattr(ds, '_sample_ds'):
        sample_ds = ds._sample_ds
        lat = sample_ds['lat']
        lon = sample_ds['lon']
    else:
        lat = ds['lat']
        lon = ds['lon']
    
    depth = xr.DataArray([0.494, 47.374], dims='depth')

    print(f"    Total timesteps à restructurer: {total_timesteps}")
    
    # Initialisation des nouvelles variables
    print("     Initialisation des variables de sortie...")
    new_vars = {
        'so': np.full((total_timesteps, 2, len(lat), len(lon)), np.nan, dtype=np.float32),
        'thetao': np.full((total_timesteps, 2, len(lat), len(lon)), np.nan, dtype=np.float32),
        'uo': np.full((total_timesteps, 2, len(lat), len(lon)), np.nan, dtype=np.float32),
        'vo': np.full((total_timesteps, 2, len(lat), len(lon)), np.nan, dtype=np.float32),
        'zos': np.full((total_timesteps, len(lat), len(lon)), np.nan, dtype=np.float32),
    }

    # MINI-CHUNKS pour éviter le crash - encore plus petit !
    chunk_size = 10  # TRÈS petit chunk : 10 timesteps à la fois
    total_chunks = (total_timesteps + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_timesteps)
        
        print(f"    Mini-chunk {chunk_idx + 1}/{total_chunks}: timesteps {start_idx}-{end_idx}")
        
        try:
            # Charger seulement ce mini-chunk
            if hasattr(ds, '_lazy_datasets'):
                data_chunk = data_generator[start_idx:end_idx]  # Le générateur charge à la demande
            else:
                data_chunk = data[start_idx:end_idx]
            
            # Restructuration de ce mini-chunk
            for ch_idx, (var_name, depth_idx) in ch_map.items():
                if depth_idx is None:
                    new_vars[var_name][start_idx:end_idx] = data_chunk[:, ch_idx, :, :]
                else:
                    new_vars[var_name][start_idx:end_idx, depth_idx, :, :] = data_chunk[:, ch_idx, :, :]
            
            # Nettoyage immédiat du chunk
            del data_chunk
            gc.collect()
            
            # Afficher la progression toutes les 10 chunks
            if (chunk_idx + 1) % 10 == 0:
                memory = psutil.virtual_memory()
                print(f"      Progression: {chunk_idx + 1}/{total_chunks} chunks, Mémoire: {memory.percent:.1f}%")
        
        except Exception as e:
            print(f"      Erreur chunk {chunk_idx + 1}: {e}")
            # Continuer avec les autres chunks
            continue

    # Nettoyage des données originales
    if not hasattr(ds, '_lazy_datasets'):
        del data
    gc.collect()
    
    print("     Création du dataset final...")

    # Créer le dataset final par variables pour économiser la mémoire
    dataset_vars = {}
    var_names = list(new_vars.keys())  # Fixer la liste avant l'itération
    
    for var_name in var_names:
        var_data = new_vars[var_name]
        print(f"      Création variable {var_name}...")
        
        if var_name == 'zos':
            dataset_vars[var_name] = xr.DataArray(
                var_data, 
                dims=('time', 'lat', 'lon'), 
                coords={'time': time, 'lat': lat, 'lon': lon}
            )
        else:
            dataset_vars[var_name] = xr.DataArray(
                var_data, 
                dims=('time', 'depth', 'lat', 'lon'), 
                coords={'time': time, 'depth': depth, 'lat': lat, 'lon': lon}
            )
        
        # Nettoyage immédiat après création
        del new_vars[var_name]
        gc.collect()
    
    # Créer le dataset final
    new_ds = xr.Dataset(dataset_vars)
    
    # Nettoyage final
    del new_vars, dataset_vars
    gc.collect()
    
    final_memory = psutil.virtual_memory()
    print(f"✅ Restructuration ULTRA-LAZY terminée! Mémoire finale: {final_memory.percent:.1f}%")
    
    return new_ds


def load_glorys_data(years: List[int], max_timesteps: int) -> xr.Dataset:
    """
    Fonction utilitaire pour charger rapidement les données GLORYS.
    Maintenant avec lazy loading automatique.
    
    Args:
        years: Années à charger
        max_timesteps: Nombre maximum de pas de temps
        
    Returns:
        Dataset restructuré prêt pour l'entraînement
    """
    if years is None:
        years = [year for year in range(1994, 2009)]
    
    print(f" Chargement GLORYS: {len(years)} années, {max_timesteps} timesteps")
    
    # Vérification mémoire préalable
    memory = psutil.virtual_memory()
    print(f" Mémoire disponible: {memory.available/1e9:.1f}GB ({100-memory.percent:.1f}%)")
    
    loader = OceanDataLoader()
    
    # Charger les données (lazy ou normal selon la mémoire)
    dataset = loader.load_subset(years, max_timesteps)
    
    # Restructurer (le lazy loading se déclenche ici)
    print(" Début de la restructuration...")
    reshaped_dataset = reshape_dataset(dataset)
    
    # Nettoyage final
    if hasattr(loader, '_lazy_datasets'):
        loader._lazy_datasets.clear()
    del dataset, loader
    gc.collect()
    
    final_memory = psutil.virtual_memory()
    print(f" Chargement terminé! Mémoire: {final_memory.percent:.1f}%")
    
    return reshaped_dataset


# Variables globales (identiques à l'original)
years = [year for year in range(1994, 2009)]


class OceanDataset:
    """
    Dataset PyTorch qui charge les données à la demande (lazy loading).
    Évite de charger tout le dataset en mémoire.
    """
    
    def __init__(self, ds, variable_names, depth_index, time_indices, 
                 stats_dict, mask_variable, robust_stats=None):
        self.ds = ds
        self.variable_names = variable_names
        self.depth_index = depth_index
        self.time_indices = time_indices
        self.stats_dict = stats_dict
        self.mask_variable = mask_variable
        self.robust_stats = robust_stats or self._compute_robust_stats()
        
        print(f" Dataset créé avec {len(time_indices)} timesteps")
        print(f" Variables: {variable_names}")
        print(f" Masque basé sur: {mask_variable}")
        
    def _compute_robust_stats(self):
        """Calcule les statistiques robustes pour la normalisation"""
        robust_stats = {}
        epsilon = 1e-8
        
        for var in self.variable_names:
            if var not in self.stats_dict or 'mean' not in self.stats_dict[var] or 'std' not in self.stats_dict[var]:
                raise ValueError(f"Statistiques manquantes pour la variable {var}")

            # fallback au calcul direct si min/max manquent
            if 'min' in self.stats_dict[var] and 'max' in self.stats_dict[var]:
                min_val = self.stats_dict[var]['min']
                max_val = self.stats_dict[var]['max']
            else:
                print(f" Calcul direct de min/max pour {var}")
                da = self.ds[var]
                if 'depth' in da.dims:
                    data = da.isel(depth=self.depth_index).values
                else:
                    data = da.values
                valid = ~np.isnan(data)
                min_val = np.nanmin(data[valid])
                max_val = np.nanmax(data[valid])
            
            center = (max_val + min_val) / 2
            scale = max((max_val - min_val) / 2, epsilon)
            
            robust_stats[var] = {
                'min': min_val,
                'max': max_val,
                'center': center,
                'scale': scale
            }
        
        return robust_stats
    
    def __len__(self):
        return len(self.time_indices)
    
    def __getitem__(self, idx):
        """
        Charge UNE SEULE carte à la demande.
        Évite de garder tout en mémoire.
        """
        t = self.time_indices[idx]
        
        # Créer le masque océan/continent
        da_mask = self.ds[self.mask_variable]
        if 'depth' in da_mask.dims:
            mask_data = da_mask.isel(time=t, depth=self.depth_index).values
        else:
            mask_data = da_mask.isel(time=t).values
        ocean_mask = ~np.isnan(mask_data)
        
        # Renforcer le masque pour toutes les variables
        for var in self.variable_names:
            da = self.ds[var]
            if 'depth' in da.dims:
                data = da.isel(time=t, depth=self.depth_index).values
            else:
                data = da.isel(time=t).values
            ocean_mask &= ~np.isnan(data)
        
        mask_tensor = torch.tensor(ocean_mask.astype(np.float32))
        
        # Traitement des variables avec normalisation robuste
        channels = []
        for var in self.variable_names:
            da = self.ds[var]
            if 'depth' in da.dims:
                data = da.isel(time=t, depth=self.depth_index).values
            else:
                data = da.isel(time=t).values
            
            # Normalisation robuste [-1,1]
            center_val = self.robust_stats[var]['center']
            scale_val = self.robust_stats[var]['scale']
            
            data_norm = data.astype(np.float32).copy()
            ocean_points = ocean_mask
            
            if np.any(ocean_points):
                ocean_values = data[ocean_points]
                normalized_ocean = (ocean_values - center_val) / scale_val
                data_norm[ocean_points] = normalized_ocean
                data_norm[~ocean_mask] = np.nan
            
            channels.append(data_norm)
        
        data_stack = np.stack(channels, axis=0)
        return torch.FloatTensor(data_stack), mask_tensor


def create_efficient_dataloaders(ds, variable_names, depth_index, time_indices, 
                                stats_dict, mask_variable, batch_size, 
                                num_workers, train_ratio=0.9, val_ratio=0.05):
    """
    Crée des dataloaders PyTorch efficaces avec workers multiples.
    
    Args:
        batch_size: Taille des batches (réduite pour économiser la mémoire)
        num_workers: Nombre de workers pour le chargement parallèle
    """
    
    # Créer le dataset
    full_dataset = OceanDataset(
        ds, variable_names, depth_index, time_indices, 
        stats_dict, mask_variable
    )
    
    # Division des indices
    total_size = len(time_indices)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    
    # Mélanger les indices
    indices = list(range(total_size))
    #random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Créer les sous-datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,  # Accélère le transfert vers GPU
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f" Dataloaders créés:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Batch size: {batch_size}, Workers: {num_workers}")
    
    return train_loader, val_loader, test_loader


# DIAGNOSTIC DU PROBLÈME TEMPOREL
# 1. DIAGNOSTIC DES INDICES TEMPORELS
def diagnose_temporal_data(ds, time_indices, train_ratio=0.9, val_ratio=0.05):
    """Diagnostique les données temporelles pour identifier le problème"""
    
    print(f" DIAGNOSTIC TEMPOREL COMPLET")
    print(f"=" * 50)
    
    # Analyse des time_indices
    total_size = len(time_indices)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    
    # Division exacte comme dans votre code
    indices = list(range(total_size))
    # SANS shuffle (comme dans votre code)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f" RÉPARTITION DES DONNÉES:")
    print(f"   - Total timesteps: {total_size}")
    print(f"   - Train indices: {len(train_indices)} (index {train_indices[0]} à {train_indices[-1]})")
    print(f"   - Val indices: {len(val_indices)} (index {val_indices[0]} à {val_indices[-1]})")
    print(f"   - Test indices: {len(test_indices)} (index {test_indices[0]} à {test_indices[-1]})")
    
    # Analyse des time_indices correspondants
    print(f"\n ANALYSE DES TIME_INDICES:")
    print(f"   - Train time_indices: {[time_indices[i] for i in train_indices[:5]]}...{[time_indices[i] for i in train_indices[-5:]]}")
    print(f"   - Val time_indices: {[time_indices[i] for i in val_indices[:5]]}...{[time_indices[i] for i in val_indices[-5:]]}")
    print(f"   - Test time_indices: {[time_indices[i] for i in test_indices[:10]]}")
    
    # Vérifier l'unicité des time_indices de test
    test_time_indices = [time_indices[i] for i in test_indices]
    unique_test_times = list(set(test_time_indices))
    
    print(f"\n UNICITÉ DES DONNÉES DE TEST:")
    print(f"   - Test indices uniques: {len(unique_test_times)}/{len(test_time_indices)}")
    
    if len(unique_test_times) < len(test_time_indices):
        print(f"    PROBLÈME DÉTECTÉ: Indices temporels dupliqués dans le test set!")
        
        # Trouver les doublons
        from collections import Counter
        counts = Counter(test_time_indices)
        duplicates = {k: v for k, v in counts.items() if v > 1}
        print(f"   📋 Doublons: {duplicates}")
    
    # Analyser les dates correspondantes
    print(f"\n DATES CORRESPONDANTES:")
    try:
        test_dates = []
        for i in test_indices[:10]:  # Premiers 10 pour diagnostic
            time_idx = time_indices[i]
            if time_idx < len(ds.time):
                date_value = ds.time[time_idx].values
                if hasattr(date_value, 'astype'):
                    date_dt = pd.to_datetime(date_value)
                    date_str = date_dt.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_value)[:10]
                test_dates.append(date_str)
                print(f"   - Index {i} → time_index {time_idx} → {date_str}")
        
        # Vérifier si les dates sont identiques
        unique_dates = list(set(test_dates))
        print(f"\n    Dates uniques dans test: {len(unique_dates)}/{len(test_dates)}")
        
        if len(unique_dates) == 1:
            print(f"    PROBLÈME MAJEUR: Toutes les dates de test sont identiques: {unique_dates[0]}")
        elif len(unique_dates) < len(test_dates):
            print(f"    PROBLÈME: Certaines dates de test sont dupliquées")
            print(f"   📋 Dates uniques: {unique_dates}")
    
    except Exception as e:
        print(f"   ❌ Erreur analyse dates: {e}")
    
    return test_indices, test_time_indices

# 2. SOLUTION AVEC ÉCHANTILLONNAGE TEMPOREL
def create_temporal_dataloaders(ds, variable_names, depth_index, time_indices, 
                               stats_dict, mask_variable, batch_size, 
                               num_workers, train_ratio=0.9, val_ratio=0.05):
    """
    Version corrigée qui assure une diversité temporelle dans le test set
    """
    
    print(f"🔧 CRÉATION DE DATALOADERS AVEC DIVERSITÉ TEMPORELLE")
    
    # Créer le dataset
    full_dataset = OceanDataset(
        ds, variable_names, depth_index, time_indices, 
        stats_dict, mask_variable
    )
    
    # S'assurer que time_indices sont uniques
    unique_time_indices = list(set(time_indices))
    unique_time_indices.sort()  # Garder l'ordre chronologique
    
    print(f"   📊 Time indices uniques: {len(unique_time_indices)}/{len(time_indices)}")
    
    if len(unique_time_indices) < len(time_indices):
        print(f"    Doublons détectés dans time_indices, utilisation des valeurs uniques")
        # Recréer le mapping
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_time_indices)}
        time_indices = unique_time_indices
    
    # Division des indices
    total_size = len(time_indices)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    
    # Mélanger pour avoir de la diversité temporelle
    indices = list(range(total_size))
    random.seed(42)  # Pour la reproductibilité
    #random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Pour les vidéos, prendre des échantillons espacés temporellement
    max_video_frames = 16  # limite maximale
    
    if len(test_indices) > max_video_frames:  # Si on a plus d'échantillons que nécessaire
        # Prendre des échantillons espacés uniformément
        step = len(test_indices) // max_video_frames
        # S'assurer qu'on prend exactement max_video_frames échantillons
        test_indices_for_videos = test_indices[::step][:max_video_frames]
        print(f"    Échantillons pour vidéos: {len(test_indices_for_videos)}/{len(test_indices)} (espacés de {step}, limité à {max_video_frames})")
    else:
        test_indices_for_videos = test_indices
        print(f"    Échantillons pour vidéos: {len(test_indices_for_videos)} (tous utilisés, < {max_video_frames})")
    
    print(f"    Indices vidéo sélectionnés: {test_indices_for_videos[:5]}...{test_indices_for_videos[-5:] if len(test_indices_for_videos) > 5 else []}")
    
    # Créer les sous-datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Dataset spécial pour vidéos avec diversité temporelle
    video_dataset = torch.utils.data.Subset(full_dataset, test_indices_for_videos)
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # ACTIVÉ pour l'entraînement
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # DataLoader spécial pour vidéos
    video_loader = DataLoader(
        video_dataset, 
        batch_size=1,  # Un échantillon à la fois pour les vidéos
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f" Dataloaders créés:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Video: {len(video_dataset)} samples, {len(video_loader)} batches")
    print(f"  Batch size: {batch_size}, Workers: {num_workers}")
    
    return train_loader, val_loader, test_loader, video_loader

#Prétraitement des données avec normalisation robuste [-1,1]
def preprocess_ocean_data_full_maps(ds, variable_names, depth_index, time_indices, 
                                  stats_dict, mask_variable, batch_size, 
                                  num_workers):
    """
    Version allégée du préprocessing qui utilise les DataLoaders PyTorch.
    Ne charge PAS tout en mémoire d'un coup.
    """
    
    print(f" Préprocessing efficace activé")
    print(f" {len(time_indices)} timesteps à traiter")
    print(f" Batch size: {batch_size}, Workers: {num_workers}")
    
    # Créer les dataloaders efficaces
    train_loader, val_loader, test_loader, video_loader = create_temporal_dataloaders(
        ds, variable_names, depth_index, time_indices, 
        stats_dict, mask_variable, batch_size, num_workers
    )
    
    # Tester un échantillon pour vérifier les dimensions
    sample_batch = next(iter(train_loader))
    data_batch, mask_batch = sample_batch
    
    print(f"📏 Dimensions d'un batch:")
    print(f"  Data: {data_batch.shape}")  # [batch_size, channels, H, W]
    print(f"  Mask: {mask_batch.shape}")  # [batch_size, H, W]
    
    return train_loader, val_loader, test_loader


# Fonction utilitaire pour calculer les statistiques robustes
def calculate_robust_stats(ds, variable_names, depth_index=None):
    """
    Calcule les statistiques min/max nécessaires pour la normalisation robuste.
    
    Args:
        ds: Dataset xarray
        variable_names: Liste des variables
        depth_index: Index de profondeur (si applicable)
    
    Returns:
        Dictionnaire des statistiques avec min/max pour chaque variable
    """
    print(" Calcul des statistiques robustes (min/max) pour normalisation [-1,1]...")
    
    stats_dict = {}
    for var in variable_names:
        print(f"  Traitement de {var}...")
        
        if 'depth' in ds[var].dims and depth_index is not None:
            values = ds[var].isel(depth=depth_index).values
        else:
            values = ds[var].values
        
        # Calculer les statistiques en ignorant les NaN
        min_val = float(np.nanmin(values))
        max_val = float(np.nanmax(values))
        mean_val = float(np.nanmean(values))
        std_val = float(np.nanstd(values))
        
        stats_dict[var] = {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val
        }
        
        print(f"    {var}: min={min_val:.3f}, max={max_val:.3f}, mean={mean_val:.3f}, std={std_val:.3f}")
    
    print(" Statistiques robustes calculées")
    return stats_dict


# ============= SAUVEGARDE AUTOMATIQUE APRÈS ENTRAÎNEMENT =============
def save_model(model, model_type, stats_dict, output_dir, training_history=None, config_params=None):
    """
    Sauvegarde automatique du modèle avec toutes les métadonnées nécessaires
    """
    model_filename = f"ocean_{model_type.lower()}_model.pth"
    model_path = os.path.join(output_dir, model_filename)
    
    # Données de base à sauvegarder
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'stats_dict': stats_dict,
        'variables': config_params.get('variables', []),
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'config': config_params or {}
    }
    
    # Ajout de l'historique d'entraînement si disponible
    if training_history:
        save_dict.update({
            'train_total_losses': training_history.get('train_total_losses', []),
            'train_recon_losses': training_history.get('train_recon_losses', []),
            'train_kl_losses': training_history.get('train_kl_losses', []),
            'val_total_losses': training_history.get('val_total_losses', []),
            'val_recon_losses': training_history.get('val_recon_losses', []),
            'val_kl_losses': training_history.get('val_kl_losses', []),
            'best_val_loss': training_history.get('best_val_loss', float('inf'))
        })
    
    # Paramètres spécifiques selon le modèle
    if model_type == "VAE" or model_type == "WAE-MMD":
        save_dict['latent_dim'] = getattr(model, 'latent_dim', config_params.get('latent_dim'))
        if model_type == "WAE-MMD":
            save_dict['lambda_reg'] = config_params.get('lambda_reg')
            save_dict['sigma_list'] = config_params.get('sigma_list')
    elif model_type == "VQVAE":
        save_dict.update({
            'num_embeddings': getattr(model, 'num_embeddings', config_params.get('num_embeddings')),
            'embedding_dim': getattr(model, 'embedding_dim', config_params.get('embedding_dim')),
            'hidden_dims': config_params.get('hidden_dims', [8, 16])
        })
    
    torch.save(save_dict, model_path)
    print(f" Modèle {model_type} sauvegardé dans {model_path}")
    return model_path

# ============= CHARGEMENT UNIVERSEL DE MODÈLE =============
def load_model_universal(model_path, device):
    """
    Charge un modèle sauvegardé et recrée l'architecture appropriée
    """
    print(f"📂 Chargement du modèle depuis {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model_type = checkpoint['model_type']
    stats_dict = checkpoint['stats_dict']
    
    # Récupération des paramètres du modèle
    if model_type == "VAE":
        from models.vae_model import OceanVAE
        model = OceanVAE(
            input_channels=checkpoint['config'].get('input_channels', 3),
            latent_dim=checkpoint['latent_dim']
        ).to(device)
    
    elif model_type == "WAE-MMD":
        from models.waemmd_model import OceanWAE
        model = OceanWAE(
            input_channels=checkpoint['config'].get('input_channels', 3),
            latent_dim=checkpoint['latent_dim'],
            hidden_dims=checkpoint['config'].get('hidden_dims', [64, 128]),
            lambda_reg=checkpoint.get('lambda_reg', 10.0),
            sigma_list=checkpoint.get('sigma_list', [1.0, 2.0, 4.0])
        ).to(device)
    
    elif model_type == "VQVAE":
        from models.vqvae_model import OceanVQVAE
        model = OceanVQVAE(
            input_channels=checkpoint['config'].get('input_channels', 3),
            embedding_dim=checkpoint['embedding_dim'],
            num_embeddings=checkpoint['num_embeddings'],
            hidden_dims=checkpoint.get('hidden_dims', [8, 16]),
            beta=checkpoint['config'].get('beta', 0.25)
        ).to(device)
    
    # Chargement des poids
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f" Modèle {model_type} chargé avec succès")
    print(f" Variables: {checkpoint.get('variables', 'Non spécifiées')}")
    print(f" Timestamp: {checkpoint.get('timestamp', 'Non spécifié')}")
    
    return model, stats_dict, checkpoint

# ============= FONCTION D'INFÉRENCE RAPIDE =============
def inference_mode(model_path, data_to_process=None):
    """
    Mode inférence rapide - charge un modèle et effectue des prédictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats_dict, checkpoint = load_model_universal(model_path, device)
    
    print(f" Modèle prêt pour l'inférence sur {device}")
    print(f" Configuration du modèle:")
    for key, value in checkpoint['config'].items():
        print(f"   - {key}: {value}")
    
    # Si des données sont fournies, effectuer l'inférence
    if data_to_process is not None:
        print("🚀 Début de l'inférence...")
        model.eval()
        with torch.no_grad():
            # Adapter selon le type de modèle
            if checkpoint['model_type'] in ["VAE", "WAE-MMD"]:
                reconstructed, mu, logvar = model(data_to_process.to(device))
                return reconstructed, mu, logvar
            elif checkpoint['model_type'] == "VQVAE":
                reconstructed, vq_loss, perplexity, encoding_indices = model(data_to_process.to(device))
                return reconstructed, vq_loss, perplexity, encoding_indices
    
    return model, stats_dict
