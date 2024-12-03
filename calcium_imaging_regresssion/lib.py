from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import os
import pickle

# Load environment variables
load_dotenv()
allen_cache_path = os.environ.get('HGMS_ALLEN_CACHE_PATH')

# Initialize BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file=str(Path(allen_cache_path) / Path('brain_observatory_manifest.json')))

# Get ophys experiments
cell_exp = boc.get_ophys_experiments(experiment_container_ids=[511511001])

# Load Transformer embeddings
file_path = "/home/maria/Documents/HuggingMouseData/TransformerEmbeddings/google_vit-base-patch16-224-in21k_embeddings.pkl"
with open(file_path, 'rb') as file:
    transfr = pickle.load(file)

stimuli = transfr['natural_movie_one']

experiment_id = 506278598
data_set_regression = boc.get_ophys_experiment_data(experiment_id)
data_set_events = boc.get_ophys_experiment_events(experiment_id)

stim_table = data_set_regression.get_stimulus_table('natural_movie_one')

def generate_event_count_vector_levels(trial_ind, neuron_ind, transformer_emb_ind):
    ts = stim_table[stim_table['repeat'] == trial_ind]['start'].values
    neuron=data_set_events[neuron_ind,ts]
    embeddings = stimuli[:,transformer_emb_ind]
    # Partition the embeddings into 5 levels based on quantiles
    quantiles = np.percentile(embeddings, [20, 40, 60, 80])
    bins = np.concatenate(([-np.inf], quantiles, [np.inf]))
    levels = np.digitize(embeddings, bins) - 1 

    event_values_per_level = []
    for level in range(5):
        indices = np.where(levels == level)[0]
        event_values = neuron[indices]
        event_values_per_level.append(event_values)

    event_values_per_level = np.array(event_values_per_level)
    event_values_nonzero=np.count_nonzero(event_values_per_level, axis=1)

    return event_values_nonzero

neuron_trial_counts=[]

for trial in range(10):
    neuron_trial_counts.append(generate_event_count_vector_levels(trial, 6, 0))

neuron_trial_counts=np.array(neuron_trial_counts)

plt.hist(neuron_trial_counts)