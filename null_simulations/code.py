from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
import torch
from dotenv import load_dotenv
from pathlib import Path
import os
import pickle
import cebra
import itertools
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader

dat=pickle.load(open('/home/maria/Documents/HuggingMouseData/TransformerEmbeddings/openai_clip-vit-base-patch32_embeddings.pkl','rb'))

def single_session_solver(data_loader, **kwargs):
    """Train a single session CEBRA model."""
    norm = True
    if kwargs['distance'] == 'euclidean':
        norm = False
    #data_loader.to(kwargs['device'])
    model = cebra.models.init(kwargs['model_architecture'], data_loader.dataset.input_dimension,
                              kwargs['num_hidden_units'],
                              kwargs['output_dimension'], norm).to(kwargs['device'])
    #data_loader.dataset.configure_for(model)
    if kwargs['distance'] == 'euclidean':
        criterion = cebra.models.InfoMSE(temperature=kwargs['temperature'])
    elif kwargs['distance'] == 'cosine':
        criterion = cebra.models.InfoNCE(temperature=kwargs['temperature'])
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), criterion.parameters()), lr=kwargs['learning_rate'])
    return cebra.solver.SingleSessionSolver(model=model,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            tqdm_on=kwargs['verbose'])

# Custom PyTorch Dataset for spike data
class SpikeDataset(Dataset):
    def __init__(self, spikes):
        # Store spikes tensor
        self.spikes = spikes
        self.input_dimension = spikes.shape[1]  

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.spikes)

    def __getitem__(self, idx):
        # Return the spike data at the specified index
        return self.spikes[idx]

@torch.no_grad()
def get_emissions(model, dataset):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)
    dataset.configure_for(model)
    return model(dataset[torch.arange(len(dataset))].to(device)).cpu().numpy()

def _compute_emissions_single(solver, dataset):
    return get_emissions(solver.model, dataset)


# Linear-Nonlinear-Poisson model
def compute_lnp_neural_activity(embeddings, neuron_weights, biases):
    """
    Compute neural activity for a population of neurons using the LNP model.

    Parameters:
    - embeddings: array of shape (num_samples, embedding_dim)
    - neuron_weights: array of shape (num_neurons, embedding_dim)
    - biases: array of shape (num_neurons,)

    Returns:
    - spikes: array of shape (num_samples, num_neurons)
    - rates: array of shape (num_samples, num_neurons)
    """
    # Linear stage: Compute the dot product between embeddings and neuron weights
    # Resulting in a matrix of shape (num_samples, num_neurons)
    linear_response = np.dot(embeddings, neuron_weights.T) + biases  # Broadcasting biases

    # Nonlinear stage: Apply exponential non-linearity to get firing rates
    rates = np.exp(linear_response)

    # Poisson spiking stage: Simulate spikes based on Poisson distribution
    spikes = np.random.poisson(rates)

    return spikes, rates

def pipeline():
    embeddings = dat['natural_movie_one']
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embedding_dim=768
    num_neurons=500
    #Generate random weight vectors for neurons (selectivity directions)
    neuron_weights = np.random.randn(num_neurons, embedding_dim)
    # Normalize the neuron weights to have unit length
    neuron_weights_norm = neuron_weights / np.linalg.norm(neuron_weights, axis=1, keepdims=True)

    # Define the bias term for each neuron (can be zeros or random)
    biases = np.zeros(num_neurons)
    spikes, rates=compute_lnp_neural_activity(embeddings_norm, neuron_weights, biases)
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    train_steps = 10000
    #ca_loader = cebra.data.ContinuousDataLoader(spikes, num_steps = train_steps, batch_size = 512, conditional = 'time_delta', time_offset =1)
    batch_size = 512
    spikes_tensor = torch.tensor(spikes, dtype=torch.float32)
    spike_dataset = SpikeDataset(spikes_tensor)
    ca_loader = DataLoader(spike_dataset, batch_size=batch_size, shuffle=True)
    cebra_ca = single_session_solver(data_loader = ca_loader, model_architecture = 'offset1-model',
                 distance = 'cosine', num_hidden_units = 128, output_dimension = 128,
                verbose = True, device = DEVICE, temperature = 1, learning_rate = 3e-4)
    cebra_ca.fit(ca_loader)
    cebra_ca_emb = _compute_emissions_single(cebra_ca, spikes)

    fig = plt.figure(figsize = (12,5))

    ax1 = plt.subplot(121)
    ax1.set_title('Ca')
    ax1.scatter(cebra_ca_emb[:,0], cebra_ca_emb[:,1], cmap = 'magma', c = np.tile(np.arange(900),9), s=1)
    ax1.axis('off')

pipeline()
