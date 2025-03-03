import torch
from torch import nn
import numpy as np
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
    def __init__(self, input_dim, n_neurons):
        super(LNPModel, self).__init__()
        self.linear = nn.Linear(input_dim, 50)  # Linear layer for each neuron
        self.linear2=nn.Linear(50, n_neurons)
    
    def forward(self, x):
        x_1 = self.linear(x)
        x_2=self.linear2(x_1)
        firing_rate = torch.exp(x_2)  # Exponential non-linearity
        return firing_rate, x_1
    
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
        for epoch in range(num_epochs):
            lnp_model.train()
            
            # Forward pass
            predicted_firing_rate = lnp_model(self.embeddings)[0].squeeze()
            
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

        return lnp_model
        

    def __call__(self, trial_index, stimulus_type='natural_movie_one_more_repeats'):
        stimulus_type='natural_movie_one_shuffled'
        stim = self.stimuli_df[self.stimuli_df['stimulus_name'] == stimulus_type]
        spike_times=self.session.spike_times
        spike_times=get_spike_intervals(spike_times,stim['start_time'].values,stim['stop_time'].values)
        spikes=torch.tensor([spike_times[key] for key in spike_times.keys()],dtype=torch.float32)[:,trial_index*900:(trial_index+1)*900].T
        '''
        lnp=LNPModel(self.embeddings.shape[1], len(spike_times.keys()))
        model=self.training_loop(lnp,spikes,trial_index)
        firing_rates, embeddings=model(self.embeddings)
        print(firing_rates.shape, embeddings.shape)
        np.save(f'./fisher_weights/firing_rates_{trial_index}.npy', firing_rates.detach().numpy(), f'./fisher_weights/firing_rates_{trial_index}.npy')
        np.save(f'./fisher_weights/embeddings_{trial_index}.npy', embeddings.detach().numpy())
        np.save(f'./fisher_weights/spikes_{trial_index}.npy', spikes.detach().numpy())
        '''
        #np.save(f'./fisher_weights/shuffled_firing_rates_{trial_index}.npy', firing_rates.detach().numpy(), f'./fisher_weights/firing_rates_{trial_index}.npy')
        #np.save(f'./fisher_weights/shuffled_embeddings_{trial_index}.npy', embeddings.detach().numpy())
        np.save(f'./fisher_weights/shuffled_spikes_{trial_index}.npy', spikes.detach().numpy())



pipeline=FrontierPipeline()
for i in range(0,10):
    pipeline(i)
