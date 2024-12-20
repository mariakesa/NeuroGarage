import torch
import torch.nn as nn
import torch.optim as optim
   
import os
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import numpy as np
import pickle

from lib_clean import get_spike_intervals

# Sample LNP Model definition
class LNPModel(nn.Module):
    def __init__(self, input_dim):
        super(LNPModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear layer for each neuron
    
    def forward(self, x):
        x = self.linear(x)
        firing_rate = torch.exp(x)  # Exponential non-linearity
        return firing_rate
    
output_dir = '/home/maria/AllenData'
manifest_path = os.path.join(output_dir, "manifest.json")

functional_conn_session=[831882777]
brain_obs_session=[757970808]

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
func_session = cache.get_session_data(brain_obs_session[0])

stimuli_df=func_session.stimulus_presentations
print(stimuli_df) 
print(stimuli_df.columns)

# Filter for the first stimulus of interest
df_one_more_repeats = stimuli_df[stimuli_df['stimulus_name'] == 'natural_scenes']

embeddings=pickle.load(open('/home/maria/Documents/HuggingMouseData/TransformerEmbeddings/google_vit-base-patch16-224-in21k_embeddings.pkl','rb'))['natural_scenes']
spike_times=func_session.spike_times
print(len(spike_times.keys()))
'''

start_times = df_one_more_repeats['start_time'].values
print(len(start_times))

stop_times = df_one_more_repeats['stop_time'].values

real_spikes=get_spike_intervals(spike_times,start_times,stop_times)
print(real_spikes)
real_spikes=real_spikes[951840133][:118]

# Assuming embeddings is a (num_samples, embedding_dim) tensor and real_spikes is (num_samples, num_neurons)
input_dim = embeddings.shape[1]
lnp_model = LNPModel(input_dim)

# Loss function (Negative Log-Likelihood) and Optimizer
criterion = nn.PoissonNLLLoss(log_input=False)  # Poisson Negative Log-Likelihood Loss
optimizer = optim.Adam(lnp_model.parameters(), lr=0.01)

# Convert data to PyTorch tensors
embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
real_spikes_tensor = torch.tensor(real_spikes, dtype=torch.float32)  # Example: Fitting to the first neuron

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    lnp_model.train()
    
    # Forward pass
    predicted_firing_rate = lnp_model(embeddings_tensor).squeeze()
    
    # Compute loss
    loss = criterion(predicted_firing_rate, real_spikes_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

import matplotlib.pyplot as plt
plt.plot(predicted_firing_rate.detach().numpy(), color='blue',alpha=0.3)
plt.plot(real_spikes_tensor.detach().numpy(), color='red',alpha=0.3)
plt.show()

plt.scatter(predicted_firing_rate.detach().numpy(),real_spikes_tensor.detach().numpy())
plt.show()

print(np.corrcoef(predicted_firing_rate.detach().numpy(),real_spikes_tensor.detach().numpy())[0,1])

from scipy.ndimage import gaussian_filter1d

sigma = 0.5
filtered_spikes = gaussian_filter1d(real_spikes_tensor.detach().numpy(), sigma)
plt.plot(predicted_firing_rate.detach().numpy()[:450], color='blue',alpha=0.3)
plt.plot(filtered_spikes[:450], color='red',alpha=0.3)
plt.show()
print(np.corrcoef(predicted_firing_rate.detach().numpy(),filtered_spikes)[0,1])'''