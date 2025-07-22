# MachineLearning-Compression

Il s'agit là d'un pipeline pour l'exploration de modèle de types auto-encodeurs variationnels appliqués à la compression de cartes océaniques au format 336 x 720 (demi degré de résolution) nous renseignant par des flottants en chaque point sur 5 variables : la salinité, la température, les courants horizontaux et verticaux, le niveau de surface maritime. Ces modèles utilise les données du jeu de données baby_glorys donnant un point de donnée par jour de 1994 à 2009, et pour un seul niveau de profondeur : la surface. 

Le pipeline comporte les étapes suivantes : chargement de données et prétraitement, entrainement, évaluation et visualisation. 
Les dossiers loss et models contiennent les fichiers de définition des modèles et fonction de perte associées à chaque modèle (VAE, WAE, VQVAE) et le dossier config contient la paramétrisation de chaque modèle, à modifier au souhait. Enfin, le notebook main consiste en la réalisation du pipeline, avec une cellule alloué au choix du modèle à utiliser, la proportion du dataset souhaitée, et à la décision d'entrainer, d'inférer, ou de charger et évaluer un modèle. 
