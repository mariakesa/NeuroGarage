import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from code2 import dat, compute_lnp_neural_activity
import pickle

#dat=pickle.load(open('/home/maria/Documents/HuggingMouseData/TransformerEmbeddings/openai_clip-vit-base-patch32_embeddings.pkl','rb'))
# Original pipeline function to generate spikes
def pipeline():
    # Assuming dat is defined somewhere
    embeddings = dat['natural_movie_one']
    embeddings_norm = embeddings #/ np.linalg.norm(embeddings, axis=1, keepdims=True)
    embedding_dim = 768
    num_neurons = 500
    
    # Generate random weight vectors for neurons (selectivity directions)
    neuron_weights = np.random.randn(num_neurons, embedding_dim)
    # Normalize the neuron weights to have unit length
    neuron_weights_norm = neuron_weights / np.linalg.norm(neuron_weights, axis=1, keepdims=True)

    # Define the bias term for each neuron (can be zeros or random)
    biases = np.zeros(num_neurons)

    # Compute spikes and rates
    spikes, rates = compute_lnp_neural_activity(embeddings_norm, neuron_weights_norm, biases)
    print(rates)
    
    return spikes

# Generate spikes from the pipeline
spikes = pipeline()

# t-SNE Visualization with time coloring
def tsne_plot_time_colored(spikes):
    # Perform t-SNE on the spike data
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    spikes_tsne = tsne.fit_transform(spikes)

    # Create time labels for the data points
    time_labels = np.arange(spikes.shape[0])  # Assuming each row represents a time point

    # Normalize time_labels to the range [0, 1] for color mapping
    time_labels_normalized = time_labels / time_labels.max()

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    plt.scatter(spikes_tsne[:, 0], spikes_tsne[:, 1], c=time_labels_normalized, cmap='viridis', alpha=0.8, s=40, edgecolor='k')
    plt.colorbar(label='Time Dimension')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Spike Data Colored by Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Create t-SNE plot of the spikes colored by time dimension
tsne_plot_time_colored(spikes)
