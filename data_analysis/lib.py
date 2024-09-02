from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.decomposition import PCA, NMF
import matplotlib.pyplot as plt

load_dotenv()

allen_cache_path = os.environ.get('HGMS_ALLEN_CACHE_PATH')
boc = BrainObservatoryCache(manifest_file=str(Path(allen_cache_path) / Path('brain_observatory_manifest.json')))
stimulus_session_dict = {
            'three_session_A': ['natural_movie_one', 'natural_movie_three'],
            'three_session_B': ['natural_movie_one'],
            'three_session_C': ['natural_movie_one', 'natural_movie_two'],
            'three_session_C2': ['natural_movie_one', 'natural_movie_two']
}
experiment_container_id=565039910


def get_n_events():
    pass

def create_nested_dict(df):
    # Group by the 'repeat' column
    grouped = df.groupby('repeat')
    
    # Create the nested dictionary
    nested_dict = {}
    for name, group in grouped:
        #print(name, group)
        nested_dict[name] = {
            'start': group['start'].tolist(),
            'end': group['end'].tolist()
        }
    
    return nested_dict

def max_pool(all_trials,start,end):
    return max(all_trials[start:end+1])

def plot_cells(experiment_container_id, experiment_type, stimulus_type):
    experiment_dict = boc.get_ophys_experiments(experiment_container_ids=[experiment_container_id])
    if experiment_type == 'three_session_C':
        for exp in experiment_dict:
            if exp['session_type'] == 'three_session_C':
                experiment_id = exp['id']
            elif exp['session_type'] == 'three_session_C2':
                experiment_id = exp['id']
    else:
        for exp in experiment_dict:
            if exp['session_type'] == experiment_type:
                experiment_id = exp['id']
    data_set_regression = boc.get_ophys_experiment_data(experiment_id)
    data_set_events= boc.get_ophys_experiment_events(experiment_id)
    cells = data_set_regression.get_cell_specimen_ids()
    cell2ix={cell:ix for ix, cell in enumerate(cells)}
    ix2cell={ix:cell for ix, cell in enumerate(cells)}
    stim_table=data_set_regression.get_stimulus_table(stimulus_type)
    stim_dict=create_nested_dict(stim_table)
    timed_array=get_cell_trials_f(get_n_events, 0, data_set_events,stim_dict)
    W, H = get_cell_nmf(timed_array)

def plot_components(H):
        # Plot the H matrix (temporal components) in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the three components in 3D space
    ax.scatter(H[0], H[1], H[2], marker='o')
    
    # Labeling the axes
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    
    ax.set_title('3D Plot of Temporal Components from NMF')
    
    plt.show()

def get_cell_nmf(timed_array):
    # Perform non-negative matrix factorization
    model = NMF(n_components=3, init='random', random_state=0)
    W = model.fit_transform(timed_array)
    H = model.components_
    print(H.shape,W.shape)
    plot_components(H)
    return W, H


def get_cell_trials_f(action_function, cell_ix, events, stim_dict):
    all_trials = events[cell_ix]
    start_times = np.array([stim_dict[j]['start'] for j in stim_dict])
    end_times = np.array([stim_dict[j]['end'] for j in stim_dict])

    # Vectorized calculation of max_pool over the specified ranges
    timed_array = np.array([[max_pool(all_trials, start, end) for start, end in zip(starts, ends)] 
                            for starts, ends in zip(start_times, end_times)])

    return timed_array

def cell_compute_n_events(experiment_id, cell_specimen_id):
    data_set = boc.get_ophys_experiment_data(experiment_id)
    #timestamps, dff = data_set.get_dff_traces(cell_spec

#natural_movie_three: 10 repeats
plot_cells(experiment_container_id, 'three_session_A', 'natural_movie_three')
