variables = "so,thetao,uo,vo,zos"
#max_timesteps = 20
patience = 25
early_stopping = True
depth_index = 0
training = True #si on entraine ou non
epochs = 2
batch_size = 16   # Ajuste selon ton GPU
num_workers = 1  # Ajuste selon ton CPU
pin_memory = True
device = "cuda"

modèle = "VQVAE" #modèle à utiliser 

#Parmamètres du modèle:
scheduler_gamma = 0.95
latent_dim = 16
embedding_dim = 16
learning_rate = 1e-4
progressive_beta = False
initial_beta=None
beta = 0.15
commitment_weight = beta
num_embeddings = 256
hidden_dims = [8, 16, 32]









