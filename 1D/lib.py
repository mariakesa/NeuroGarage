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

class LNPModel(nn.Module):
    def __init__(self, n_neurons):
        super(LNPModel, self).__init__()
        self.linear = nn.Linear(1, n_neurons)  # Linear layer for each neuron
    
    def forward(self, x):
        linear_output = self.linear(x)  # Compute w x_t
        firing_rate = torch.exp(linear_output)  # Exponential non-linearity
        return firing_rate, linear_output  # Return both for loss computation

class FrontierPipeline:
    def __init__(self, session_id=831882777):
        output_dir = os.environ['ALLEN_NEUROPIXELS_PATH']
        manifest_path = os.path.join(output_dir, "manifest.json")
        self.cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
        self.session = self.cache.get_session_data(session_id)
        self.stimuli_df = self.session.stimulus_presentations
        embeddings = pickle.load(open('/home/maria/Documents/HuggingMouseData/TransformerEmbeddings/openai_clip-vit-base-patch32_embeddings.pkl', 'rb'))['natural_movie_one']
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)[:,0].unsqueeze(1)

    def training_loop(self, lnp_model, real_spikes_tensor):
        # Training loop
        # Manually compute the negative log-likelihood loss
        optimizer = optim.Adam(lnp_model.parameters(), lr=0.001, weight_decay=1e-2)
        num_epochs = 10000
        delta = 0.0333  # Time bin duration
        ln_delta = torch.log(torch.tensor(delta))

        for epoch in range(num_epochs):
            lnp_model.train()
            
            # Forward pass
            predicted_firing_rate, linear_output = lnp_model(self.embeddings)
            predicted_firing_rate = predicted_firing_rate.squeeze()  # Shape: (n_samples, n_neurons)
            linear_output = linear_output.squeeze()  # Shape: (n_samples, n_neurons)
            
            # Compute loss manually
            # Loss = sum_t [e^{w x_t} * delta - w x_t * y_t]
            loss = torch.sum(predicted_firing_rate * delta - real_spikes_tensor * linear_output)
                        # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print loss
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        print('Training finished')
        weights = lnp_model.linear.weight.detach().numpy()
        biases = lnp_model.linear.bias.detach().numpy()
        return lnp_model, weights, biases
    
    def __call__(self, trial_index, stimulus_type='natural_movie_one_more_repeats'):
        stim = self.stimuli_df[self.stimuli_df['stimulus_name'] == stimulus_type]
        spike_times = self.session.spike_times
        # Assuming get_spike_intervals is defined elsewhere
        spike_times = get_spike_intervals(spike_times, stim['start_time'].values, stim['stop_time'].values)
        spikes = torch.tensor([spike_times[key] for key in spike_times.keys()], dtype=torch.float32)[:, trial_index*900:(trial_index+1)*900].T
        lnp = LNPModel(len(spike_times.keys()))
        lnp_model, weights, biases = self.training_loop(lnp, spikes)

        return lnp_model, weights, biases, self.embeddings

pipeline = FrontierPipeline()
pipeline(0)