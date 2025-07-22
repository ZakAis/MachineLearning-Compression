variables = "so,thetao,uo,vo,zos"
#max_timesteps = 20
patience = 10
early_stopping = True
depth_index = 0 
training = True  #si on entraine ou non
epochs = 2
batch_size = 16       # Ajuste selon ton GPU
num_workers = 8    # Ajuste selon ton CPU
pin_memory = True
device = "cuda"

modèle = "VAE" #modèle à utiliser 

#Parmamètres du modèle:
scheduler_gamma = 0.95
latent_dim = 16
learning_rate = 1e-3
progressive_beta = False
initial_beta=1e-7
beta = 1e-7









