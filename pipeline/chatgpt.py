import torch

def compute_block_fisher_information(model, data_loader, n_neurons):
    model.eval()
    fisher_info_blocks = [None] * n_neurons
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.requires_grad_(True)
        firing_rate = model(batch_x)
        
        for neuron_idx in range(n_neurons):
            # Compute log-likelihood for each neuron separately
            log_likelihood = torch.sum(batch_y[:, neuron_idx] * torch.log(firing_rate[:, neuron_idx] + 1e-10) - firing_rate[:, neuron_idx])
            
            # Compute gradients
            grads = torch.autograd.grad(log_likelihood, model.parameters(), retain_graph=True, create_graph=True)
            grads = torch.cat([g.contiguous().view(-1) for g in grads])
            
            # Accumulate outer product
            if fisher_info_blocks[neuron_idx] is None:
                fisher_info_blocks[neuron_idx] = torch.ger(grads, grads)
            else:
                fisher_info_blocks[neuron_idx] += torch.ger(grads, grads)
    
    # Average and invert each block
    for neuron_idx in range(n_neurons):
        fisher_info_blocks[neuron_idx] /= len(data_loader)
        fisher_info_blocks[neuron_idx] += 1e-6 * torch.eye(fisher_info_blocks[neuron_idx].size(0))
        fisher_info_blocks[neuron_idx] = torch.inverse(fisher_info_blocks[neuron_idx])
    
    return fisher_info_blocks

# Example usage
fisher_info_blocks = compute_block_fisher_information(model, data_loader, n_neurons=10)

# Extract variances for each neuron
variances_per_neuron = [torch.diag(block) for block in fisher_info_blocks]
for idx, variances in enumerate(variances_per_neuron):
    print(f"Neuron {idx+1} variances:", variances)
