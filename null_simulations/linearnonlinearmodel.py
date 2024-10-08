import torch
import torch.nn as nn
import torch.optim as optim

# Sample LNP Model definition
class LNPModel(nn.Module):
    def __init__(self, input_dim):
        super(LNPModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear layer for each neuron
    
    def forward(self, x):
        x = self.linear(x)
        firing_rate = torch.exp(x)  # Exponential non-linearity
        return firing_rate

# Assuming embeddings is a (num_samples, embedding_dim) tensor and real_spikes is (num_samples, num_neurons)
input_dim = embeddings.shape[1]
lnp_model = LNPModel(input_dim)

# Loss function (Negative Log-Likelihood) and Optimizer
criterion = nn.PoissonNLLLoss(log_input=False)  # Poisson Negative Log-Likelihood Loss
optimizer = optim.Adam(lnp_model.parameters(), lr=0.01)

# Convert data to PyTorch tensors
embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
real_spikes_tensor = torch.tensor(real_spikes[:, 0], dtype=torch.float32)  # Example: Fitting to the first neuron

# Training loop
num_epochs = 100
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
