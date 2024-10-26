import torch
from torch.distributions import Categorical, Poisson

def generate_data(rates, mixture, T):
    rates = torch.tensor(rates, dtype=torch.float32)

    cat = Categorical(torch.tensor(mixture, dtype=torch.float32))
    T = 1000

    zs = cat.sample(sample_shape=(T,))

    # Sample the spike counts from a Poisson distribution 
    # using the rate for the corresponding state.
    # Note: this uses PyTorch's broadcasting semantics.
    xs = Poisson(rates[zs]).sample()

    return xs

def generate_gamma_prior(alpha, beta):
    return torch.tensor(alpha, dtype=torch.float32), torch.tensor(beta, dtype=torch.float32)

def EM_algorithm(xs, alpha, beta):
    rate_guess = torch.tensor([0.1, 1.0], dtype=torch.float32)
    for i in range(1000):
        up_pmf = torch.exp(Poisson(rate_guess[1]).log_prob(xs))
        down_pmf = torch.exp(Poisson(rate_guess[0]).log_prob(xs))
        
        z = up_pmf < down_pmf

        # Compute the posterior mean of the rates
        alpha[0] = alpha[0] + xs[z].sum()
        beta[0] = beta[0] + z.sum()

        alpha[1] = alpha[1] + xs[~z].sum()
        beta[1] = beta[1] + (~z).sum()

        rate_guess = (alpha - 1) / beta

        print(rate_guess)
    return rate_guess
    

def pipeline():
    rates = [1.0, 10.0]
    mixture = [0.5, 0.5]
    T = 1000
    xs = generate_data(rates, mixture, T)
    alpha, beta = generate_gamma_prior([10.0, 5.0], [1.0, 1.0])
    estimates = EM_algorithm(xs, alpha, beta)

pipeline()
