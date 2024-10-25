import torch
from typing import Dict

config_dict = {
    'K': 10,
    'T': 100,
    'N': 286,
    'sigma': 10.0,
    'lambda': 1.0,
    'amplitude_threshold': 1.0,
    }

def normalize_vec(weights: torch.Tensor) -> torch.Tensor:
    '''
    This is inefficient, but I like it.
    '''

    # Calculate the Gram matrix: weights.T @ weights
    gram_matrix = weights.T @ weights

    # Extract the diagonal elements (which are the squared column norms)
    squared_norms = gram_matrix.diag()

    # Take the square root of the diagonal elements to get the column norms
    column_lengths = torch.sqrt(squared_norms)

    # Normalize each column by its respective length
    normalized_weights = weights / column_lengths

    return normalized_weights

def make_data_(config_dict: Dict[str, float]) -> None:
    K = config_dict.get('K', 10)  # Provide default values if necessary
    T = config_dict.get('T', 100)
    N = config_dict.get('N', 286)
    sigma = config_dict.get('sigma', 1.0)
    lambda_ = config_dict.get('lambda', 1.0)  # Avoid using 'lambda' as it's a reserved keyword
    amplitude_threshold = config_dict.get('amplitude_threshold', 1.0)
    # Create an Exponential distribution with rate parameter 1.0

    distribution = torch.distributions.exponential.Exponential(lambda_)
    amplitudes = distribution.sample((T, K))
    #amplitudes[amplitudes<amplitude_threshold]=0

    weights = torch.distributions.Uniform(0, 1).sample((N, K))
    weights = normalize_vec(weights)

    gaussian_noise = torch.randn(N, T) * sigma

    # Generate the data
    data=weights @ amplitudes.T + gaussian_noise

    return data

import torch
from torch.optim import Adam
from typing import Dict

# Define the data generation function with parameters as tensors
def make_data(K, T, N, lambda_, sigma, amplitude_threshold):
    # Define the Exponential distribution with lambda_
    distribution = torch.distributions.exponential.Exponential(lambda_)
    amplitudes = distribution.rsample((T, K))  # Reparameterized sampling

    # Apply smooth thresholding
    amplitudes = amplitudes * torch.sigmoid(10 * (amplitudes - amplitude_threshold))

    # Generate weights and normalize
    weights = torch.distributions.Uniform(0, 1).sample((N, K))
    weights = weights / torch.norm(weights, dim=0, keepdim=True)

    # Generate Gaussian noise
    gaussian_noise = torch.randn(N, T) * sigma

    # Compute data
    data = weights @ amplitudes.T + gaussian_noise
    return data

import torch
from torch.optim import Adam
from typing import Dict

# Define the data generation function with parameters as tensors
def make_data(K, T, N, lambda_, sigma):
    # Define the Exponential distribution with lambda_
    distribution = torch.distributions.exponential.Exponential(lambda_)
    amplitudes = distribution.rsample((T, K))  # Reparameterized sampling

    # Apply smooth thresholding
    #amplitudes = amplitudes * torch.sigmoid(10 * (amplitudes - amplitude_threshold))

    # Generate weights and normalize
    weights = torch.distributions.Uniform(0, 1).sample((N, K))
    weights = weights / torch.norm(weights, dim=0, keepdim=True)

    
    # Use sigma squared to ensure positivity
    sigma_squared = sigma

    # Generate Gaussian noise
    gaussian_noise = torch.distributions.Normal(0, sigma_squared).sample((N, T))
    # Generate Gaussian noise
    #gaussian_noise = torch.distributions.Normal(0, sigma_squared).rsample((N, T))


    # Compute data
    data = weights @ amplitudes.T + gaussian_noise
    return data

target_data=make_data_(config_dict)

# Define learnable parameters
K, T, N = 10, 100, 286
lambda_ = torch.tensor(10.0, requires_grad=True)
sigma = torch.tensor(0.1, requires_grad=True)
#amplitude_threshold = torch.tensor(2.0, requires_grad=True)

# Define target or cost function
target_data = torch.randn(N, T)  # Hypothetical target matrix

# Define an optimizer
#optimizer = Adam([lambda_, sigma, amplitude_threshold], lr=0.01)
optimizer = Adam([lambda_, sigma], lr=0.01)
optimizer = Adam([lambda_], lr=0.01)

# Training loop
for step in range(10000):
    # Generate data with the current parameters
    data = make_data(K, T, N, lambda_, sigma)

    # Define the cost function (e.g., mean squared error with target data)
    reg_lambda = 0.1 * (lambda_ ** 2)
    reg_sigma = 0.1 * (sigma ** 2)
    loss = torch.nn.functional.mse_loss(data, target_data) + reg_lambda


    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        #print(f"Step {step}, Loss: {loss.item()}, Lambda: {lambda_.item()}, Sigma: {sigma.item()}, Threshold: {amplitude_threshold.item()}")
        print(f"Step {step}, Loss: {loss.item()}, Lambda: {lambda_.item()}, Sigma: {sigma.item()}")
