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

class Loader(DataLoader):
    def __init__(self, session_id, stimulus, shuffled=False, single_trial_limit=True, fold_index=None):
        self.shuffled = shuffled
        self.single_trial_limit = single_trial_limit
        self.fold_index = fold_index 
        output_dir=os.environ['ALLEN_NEUROPIXELS_PATH']
        manifest_path = os.path.join(output_dir, "manifest.json")
        self.cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
        self.session= self.cache.get_session_data(session_id)
        self.stimulus=stimulus
        self.stimuli_df=self.session.stimulus_presentations
        if stimulus=='natural_scenes':
            self.n_splits=4
        train_index, test_index = self.make_splits()

    def make_splits(self):
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffled)
        if self.stimulus=='natural_scenes':
            n_data_points=118
        for i, (train_index, test_index) in enumerate(kf.split(np.arange(n_data_points))):
            if i==self.fold_index:
                return train_index, test_index

    
    def __len__(self):
        if self.stimulus=='natural_scenes':
            return 4
    
    def __getitem__(self, idx):
        return self.dataset[idx]

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


def train_test_split_interleaved(self, movie_stim_table, dff_traces, trial, embedding, test_set_size):
    '''
    From https://github.com/MouseLand/rastermap/blob/main/notebooks/tutorial.ipynb
    '''
    stimuli = movie_stim_table.loc[movie_stim_table['repeat'] == trial]
    n_time = stimuli.shape[0]
    n_segs = 20
    n_len = n_time / n_segs
    sinds = np.linspace(0, n_time - n_len, n_segs).astype(int)
    itest = (sinds[:, np.newaxis] +
                np.arange(0, n_len * test_set_size, 1, int)).flatten()
    itrain = np.ones(n_time, "bool")
    itrain[itest] = 0
    itest = ~itrain
    train_inds = stimuli['start'].values[itrain]
    test_inds = stimuli['start'].values[itest]
    y_train = dff_traces[:, train_inds]
    y_test = dff_traces[:, test_inds]
    X_train = embedding[itrain]
    X_test = embedding[itest]
    return {'y_train': y_train, 'y_test': y_test, 'X_train': X_train, 'X_test': X_test}

# Sample LNP Model definition
class LNPModel(nn.Module):
    def __init__(self, input_dim, n_neurons):
        super(LNPModel, self).__init__()
        self.linear = nn.Linear(input_dim, n_neurons)  # Linear layer for each neuron
    
    def forward(self, x):
        x = self.linear(x)
        firing_rate = torch.exp(x)  # Exponential non-linearity
        return firing_rate

class FrontierPipeline:
    def __init__(self, session_id=831882777):
        output_dir=os.environ['ALLEN_NEUROPIXELS_PATH']
        manifest_path = os.path.join(output_dir, "manifest.json")
        self.cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
        self.session= self.cache.get_session_data(session_id)
        self.stimuli_df=self.session.stimulus_presentations
        embeddings=pickle.load(open('/home/maria/Documents/HuggingMouseData/TransformerEmbeddings/google_vit-base-patch16-224-in21k_embeddings.pkl','rb'))['natural_movie_one']
        self.embeddings=torch.tensor(embeddings, dtype=torch.float32)

    def training_loop(self, lnp_model, real_spikes_tensor, trial_index):
        # Training loop
        # Loss function (Negative Log-Likelihood) and Optimizer
        criterion = nn.PoissonNLLLoss(log_input=False)  # Poisson Negative Log-Likelihood Loss
        optimizer = optim.Adam(lnp_model.parameters(), lr=0.01)
        num_epochs = 10000
        for epoch in range(num_epochs):
            lnp_model.train()
            
            # Forward pass
            predicted_firing_rate = lnp_model(self.embeddings).squeeze()
            
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
        print(lnp_model.linear.weight.shape)
        weights=lnp_model.linear.weight.detach().numpy()
        np.save(f'./weights_to_analyze/weights_{trial_index}.npy',weights)  

    def __call__(self, trial_index, stimulus_type='natural_movie_one_more_repeats'):
        stim = self.stimuli_df[self.stimuli_df['stimulus_name'] == stimulus_type]
        spike_times=self.session.spike_times
        spike_times=get_spike_intervals(spike_times,stim['start_time'].values,stim['stop_time'].values)
        spikes=torch.tensor([spike_times[key] for key in spike_times.keys()],dtype=torch.float32)[:,trial_index*900:(trial_index+1)*900].T
        print(spikes.shape)
        lnp=LNPModel(self.embeddings.shape[1], len(spike_times.keys()))
        print(self.embeddings.shape)   
        print(lnp(self.embeddings).shape)
        self.training_loop(lnp,spikes,trial_index)


def compute_fisher_information(model, data_loader):
    model.eval()
    fisher_info = None
    for x, y in data_loader:
        x = x.requires_grad_(True)
        firing_rate = model(x)
        log_likelihood = torch.sum(y * torch.log(firing_rate) - firing_rate - torch.lgamma(y + 1))
        grads = torch.autograd.grad(log_likelihood, model.parameters(), create_graph=True)
        grads = torch.cat([g.view(-1) for g in grads])
        fisher_info += torch.ger(grads, grads)
    fisher_info /= len(data_loader)
    return fisher_info.inverse()


pipeline=FrontierPipeline()
for i in range(60):
    pipeline(i)

        