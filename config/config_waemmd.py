variables = "so,thetao,uo,vo,zos"
input_channels = 5
patience = 10
early_stopping = True
depth_index = 0
training = True #si on entraine ou non
epochs = 2
batch_size = 16        # Ajuste selon ton GPU
num_workers = 1      # Ajuste selon ton CPU
pin_memory = True
device = "cuda"

#Parmamètres du modèle:
scheduler_gamma = 0.95
latent_dim = 16
learning_rate = 1e-4
progressive_beta = False
initial_beta=100
beta = 100
lambda_reg = 10
sigma_list = [1.0, 5.0, 25.0]
hidden_dims = [64, 128, 256]
