import torch
from torch.distributions import Categorical, Poisson

def generate_data(rates, mixture, T):
    rates = torch.tensor(rates, dtype=torch.float32)

    cat = Categorical(torch.tensor(mixture, dtype=torch.float32))
    zs = cat.sample(sample_shape=(T,))

    # Sample the spike counts from a Poisson distribution 
    xs = Poisson(rates[zs]).sample()

    return xs

def generate_gamma_prior(alpha, beta):
    return torch.tensor(alpha, dtype=torch.float32), torch.tensor(beta, dtype=torch.float32)

def EM_algorithm(xs, alpha, beta, mixture):
    rate_guess = torch.tensor([0.1, 1.0], dtype=torch.float32)
    pi = torch.tensor(mixture, dtype=torch.float32)  # Initial mixture weights

    for i in range(1000):
        # E-step: Calculate the responsibilities
        up_pmf = torch.exp(Poisson(rate_guess[1]).log_prob(xs)) * pi[1]
        down_pmf = torch.exp(Poisson(rate_guess[0]).log_prob(xs)) * pi[0]

        responsibility_up = up_pmf / (up_pmf + down_pmf)
        responsibility_down = down_pmf / (up_pmf + down_pmf)

        # M-step: Update the mixture weights and rates
        pi[0] = responsibility_down.mean()
        pi[1] = responsibility_up.mean()

        alpha[0] = alpha[0] + (responsibility_down * xs).sum()
        beta[0] = beta[0] + responsibility_down.sum()

        alpha[1] = alpha[1] + (responsibility_up * xs).sum()
        beta[1] = beta[1] + responsibility_up.sum()

        rate_guess = (alpha - 1) / beta

        print(f"Iteration {i}: rate_guess = {rate_guess}, pi = {pi}")

    return rate_guess, pi

def pipeline():
    rates = [1.0, 10.0]
    mixture = [0.7, 0.3]  # Unequal mixture weights
    T = 1000
    xs = generate_data(rates, mixture, T)
    alpha, beta = generate_gamma_prior([10.0, 5.0], [1.0, 1.0])
    estimates, mixture_estimates = EM_algorithm(xs, alpha, beta, mixture)

pipeline()
