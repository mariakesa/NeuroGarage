import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import wavfile

# Function to parse spike file
def parse_spike_file(filename):
    with open(filename, 'r') as file:
        data = [list(map(float, line.strip().split())) for line in file if line.strip()]
    return data

# Function to bin spike times
def bin_spike_times(spike_array, bin_size_ms=100):
    max_time = 2000
    bins = np.arange(0, max_time + bin_size_ms, bin_size_ms)
    binned_spike_counts = np.zeros((len(spike_array), len(bins) - 1), dtype=int)
    for i, spikes in enumerate(spike_array):
        binned_spike_counts[i], _ = np.histogram(spikes, bins=bins)
    return binned_spike_counts, bins

# Function to load audio file
def load_audio_file(file_path):
    sample_rate, data = wavfile.read(file_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return sample_rate, data

# Load and process spike and stimulus data
spks = parse_spike_file('../crcns-student/data/l2a_avg/conspecific/spike1')
spike_counts, bins = bin_spike_times(spks, bin_size_ms=100)
spike_counts=np.mean(spike_counts, axis=0).reshape(1, -1)
file_path = "../crcns-student/data/all_stims/D54ABC42488F995C789F351A34316039.wav"
sample_rate, stimulus = load_audio_file(file_path)
stimulus = stimulus.reshape(-1, 1)
print(stimulus.shape)
print(spike_counts.shape)

# Dataset class
class SpikeDataset(Dataset):
    def __init__(self, stimulus, spike_counts, window_size=2800):
        self.stimulus = stimulus  # Shape: (num_timepoints, 1)
        self.spike_counts = spike_counts  # Shape: (num_trials, num_bins)
        self.window_size = window_size
        
        # Prepare the stimulus windows and corresponding spike counts
        self.inputs = []
        self.outputs = []
        num_trials, num_bins = spike_counts.shape
        
        for trial_idx in range(num_trials):
            for bin_idx in range(1, num_bins):
                stim_start_idx = bin_idx * (window_size-1)
                stim_end_idx = stim_start_idx + window_size
                if stim_end_idx > len(stimulus):
                    continue
                stimulus_window = stimulus[stim_start_idx:stim_end_idx]
                self.inputs.append(stimulus_window)
                self.outputs.append(spike_counts[trial_idx, bin_idx])
        
        # Convert to tensors
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32).squeeze(-1)
        self.outputs = torch.tensor(self.outputs, dtype=torch.float32)
        print(self.outputs.shape, self.inputs.shape)
    
    def __len__(self):
        return len(self.outputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# Define the Poisson Linear Nonlinear model
class PoissonLNModel(nn.Module):
    def __init__(self, input_size):
        super(PoissonLNModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.nonlinear = nn.Softplus()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x

# Instantiate dataset, dataloader, model, optimizer, and loss function
dataset = SpikeDataset(stimulus, spike_counts, window_size=2500)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = PoissonLNModel(input_size=dataset.inputs.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.PoissonNLLLoss(log_input=False)
print(dataset.inputs.shape[1])

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs = inputs.view(inputs.size(0), -1)  # Ensure input accounts for batch dimension
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

print(model.linear.weight)
np.save('model_weights.npy', model.linear.weight.detach().numpy())


