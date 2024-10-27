import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import os
from dotenv import load_dotenv
from torch import nn
import torch
import pickle
import torch.optim as optim
from torch.autograd import grad
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

load_dotenv()

def get_spike_intervals(spike_times, start_times, stop_times):
    spikes_in_intervals = {}

    # Convert start_times and stop_times to numpy arrays for faster operations
    start_times = np.array(start_times)
    stop_times = np.array(stop_times)

    print(start_times-stop_times)

    # Loop through each neuron, but optimize the spike finding with vectorized operations
    for neuron_id, times in spike_times.items():
        # Convert times to a numpy array if it's not already
        times = np.array(times)
        
        # Use numpy's searchsorted to find the indices where the start and stop times would fit
        start_indices = np.searchsorted(times, start_times, side='left')
        stop_indices = np.searchsorted(times, stop_times, side='right')
        
        # Get the number of spikes in each interval by subtracting indices
        spikes_in_intervals[neuron_id] = stop_indices - start_indices
    
    return spikes_in_intervals


# Sample LNP Model definition
class LNPModel(nn.Module):
    def __init__(self, input_dim, n_neurons):
        super(LNPModel, self).__init__()
        self.linear = nn.Linear(input_dim, 10)  # Linear layer for each neuron
        self.linear2=nn.Linear(10, n_neurons)
    
    def forward(self, x):
        x_0 = self.linear(x)
        x=self.linear2(x_0)
        firing_rate = torch.exp(x)  # Exponential non-linearity
        return firing_rate, x_0

class FrontierPipeline:
    def __init__(self, session_id=831882777):
        output_dir=os.environ['ALLEN_NEUROPIXELS_PATH']
        manifest_path = os.path.join(output_dir, "manifest.json")
        self.cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
        self.session= self.cache.get_session_data(session_id)
        self.stimuli_df=self.session.stimulus_presentations
        embeddings=pickle.load(open('/home/maria/Documents/HuggingMouseData/TransformerEmbeddings/openai_clip-vit-base-patch32_embeddings.pkl','rb'))['natural_movie_one']
        self.embeddings=torch.tensor(embeddings, dtype=torch.float32)

    def training_loop(self, lnp_model, real_spikes_tensor, trial_index):
        # Training loop
        # Loss function (Negative Log-Likelihood) and Optimizer
        criterion = nn.PoissonNLLLoss(log_input=False)  # Poisson Negative Log-Likelihood Loss
        optimizer = optim.Adam(lnp_model.parameters(), lr=0.001, weight_decay=1e-2)
        num_epochs = 10000
        delta=0.0333
        for epoch in range(num_epochs):
            lnp_model.train()
            
            # Forward pass
            predicted_firing_rate, _ = lnp_model(self.embeddings)
            predicted_firing_rate=predicted_firing_rate.squeeze()*delta
            # Compute loss
            loss = criterion(predicted_firing_rate, real_spikes_tensor)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print loss
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        print('Training finished')
        weights=lnp_model.linear2.weight.detach().numpy()
        return lnp_model, weights
    
    def __call__(self, trial_index, stimulus_type='natural_movie_one_more_repeats'):
        stim = self.stimuli_df[self.stimuli_df['stimulus_name'] == stimulus_type]
        spike_times=self.session.spike_times
        spike_times=get_spike_intervals(spike_times,stim['start_time'].values,stim['stop_time'].values)
        spikes=torch.tensor([spike_times[key] for key in spike_times.keys()],dtype=torch.float32)[:,trial_index*900:(trial_index+1)*900].T
        lnp=LNPModel(self.embeddings.shape[1], len(spike_times.keys()))
        lnp_model, weights=self.training_loop(lnp,spikes,trial_index)

        return lnp_model, weights, self.embeddings

def FisherInformation(lnp_model, embeddings, neuron_index=0, delta=0.0333):
    """
    Calculate the Fisher Information Matrix using vectorized operations.

    Parameters:
    - lnp_model: The Linear Nonlinear Poisson model.
    - embeddings: The input embeddings (torch tensor).
    - neuron_index: Index of the neuron to calculate Fisher information for.
    - delta: Scaling factor (e.g., time window).

    Returns:
    - fisher_matrix: The Fisher Information Matrix (numpy array).
    """
    # Compute firing rates using the LNP model
    firing_rate, embeddings = lnp_model(embeddings)  # Shape: (n_observations, n_neurons)
    print("Firing rate shape:", firing_rate.shape)
    
    # Select the neuron of interest
    take_neuron = firing_rate[:, neuron_index].detach().numpy()
    print(f"take_neuron_{neuron_index} shape:", take_neuron.shape)
    print(f"take_neuron_{neuron_index} stats: min =", take_neuron.min(), 
          "max =", take_neuron.max(), "mean =", take_neuron.mean())
    
    
    # Ensure firing rates are positive
    if np.any(take_neuron < 0):
        raise ValueError(f"Firing rates for neuron {neuron_index} contain negative values.")
    
    # Apply scaling if necessary
    scaled_firing_rate = take_neuron * delta
    print("Scaled firing rate stats: min =", scaled_firing_rate.min(), 
          "max =", scaled_firing_rate.max(), "mean =", scaled_firing_rate.mean())
    
    # Ensure scaled_firing_rate remains positive
    if np.any(scaled_firing_rate < 0):
        raise ValueError("Negative scaled firing rates detected after applying delta.")
    
    # Convert embeddings to NumPy array
    X = embeddings.detach().numpy()  # Shape: (n_observations, n_features)
    
    # Compute scaled embeddings
    scaled_X = X * scaled_firing_rate[:, np.newaxis]  # Shape: (n_observations, n_features)
    
    # Compute Fisher information matrix
    fisher_matrix = scaled_X.T @ X  # Shape: (n_features, n_features)
    
    print("Fisher information matrix shape:", fisher_matrix.shape)
    print("Fisher information matrix stats: min =", fisher_matrix.min(), 
          "max =", fisher_matrix.max(), "mean =", fisher_matrix.mean())
    
    # Check if Fisher matrix is positive semi-definite
    eigenvalues = np.linalg.eigvalsh(fisher_matrix)
    if np.any(eigenvalues < -1e-8):  # Allowing for small numerical errors
        print("Warning: Fisher information matrix has negative eigenvalues.")
    else:
        print("Fisher information matrix is positive semi-definite.")
    
    return fisher_matrix



#pipeline=FrontierPipeline()
#for i in range(1):
    #pipeline(i)