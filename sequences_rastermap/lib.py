import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from dotenv import load_dotenv
import numpy as np
from rastermap import Rastermap
from scipy.ndimage import gaussian_filter1d

load_dotenv()

stimulus_repeats_dict={'natural_movie_one_more_repeats': 60}
stimulus_length_dict={'natural_movie_one_more_repeats': 900}    

def get_spike_intervals(spike_times, start_times, stop_times):
    spikes_in_intervals = {}

    # Convert start_times and stop_times to numpy arrays for faster operations
    start_times = np.array(start_times)
    stop_times = np.array(stop_times)

    # Loop through each neuron, but optimize the spike finding with vectorized operations
    for neuron_id, times in spike_times.items():
        # Convert times to a numpy array if it's not already
        times = np.array(times)
        
        # Use numpy's searchsorted to find the indices where the start and stop times would fit
        start_indices = np.searchsorted(times, start_times, side='left')
        stop_indices = np.searchsorted(times, stop_times, side='right')
        
        # Get the number of spikes in each interval by subtracting indices
        spikes_in_intervals[neuron_id] = stop_indices - start_indices
    
    return spikes_in_intervals

def get_session(session_id):
    data_dir=os.environ['ALLEN_NEUROPIXELS_PATH']
    manifest_path = os.path.join(data_dir, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    session= cache.get_session_data(session_id)
    return session

def get_context(stimulus_type, session):
    stimuli_df=session.stimulus_presentations
    stim = stimuli_df[stimuli_df['stimulus_name'] == stimulus_type]
    spike_times = session.spike_times
    return stim, spike_times

def get_spikes(spike_times, stim, trial_index, stimulus_type):
    spike_times=get_spike_intervals(spike_times,stim['start_time'].values,stim['stop_time'].values)
    spikes=np.array([spike_times[key] for key in spike_times.keys()])[:,trial_index*stimulus_length_dict[stimulus_type]:(trial_index+1)*stimulus_length_dict[stimulus_type]]
    all_zero_neurons=np.where(np.sum(spikes, axis=1)==0)[0]
    return spikes, all_zero_neurons


def rastermap_pipeline(stimulus_type='natural_movie_one_more_repeats', session_id=831882777):
    session=get_session(session_id)
    stim, spike_times = get_context(stimulus_type, session)
    n_trials=stimulus_repeats_dict[stimulus_type]
    #n_trials=1
    n_trials=2
    all_zero_neuron_dct={}
    spikes_dct={}
    for trial in range(n_trials):
        spikes, all_zero_neurons=get_spikes(spike_times, stim, trial, stimulus_type)
        all_zero_neuron_dct[trial]=all_zero_neurons
        spikes_dct[trial]=spikes.astype("float32")
    for trial_1 in range(n_trials):
        for trial_2 in range(n_trials):
            if trial_1<trial_2:
                neurons_to_exclude=list(set(all_zero_neuron_dct[trial_1]).union(set(all_zero_neuron_dct[trial_2])))
                exclude_mask = np.isin(np.arange(spikes_dct[trial_1].shape[0]), neurons_to_exclude)

                # Apply the same exclusion mask to both trials
                trial1_spks = spikes_dct[trial_1][~exclude_mask]
                trial2_spks = spikes_dct[trial_2][~exclude_mask]
                
                sigma = 10.0 

                # Apply Gaussian smoothing along the time axis (axis=1) for each neuron.
                trial1_spks = gaussian_filter1d(trial1_spks, sigma=sigma, axis=1)
                trial2_spks = gaussian_filter1d(trial2_spks, sigma=sigma, axis=1)

                model1 = Rastermap(n_PCs=200, n_clusters=50, 
                  locality=0.9, time_lag_window=50).fit(trial1_spks)
                isort1 = model1.isort

                model2 = Rastermap(n_PCs=200, n_clusters=50, 
                  locality=0.9, time_lag_window=50).fit(trial2_spks)
                isort2 = model2.isort

                print(alignment_algorithm(isort1, isort2))

                np.save('isort1.npy', isort1)
                np.save('isort2.npy', isort2)   



def alignment_algorithm(seq1, seq2, window_len=4):
    def hash_windows(seq, window_len):
        dct = {tuple(sorted(seq[i:i+window_len])): i for i in range(len(seq) - window_len + 1)}
        return dct
    seq1_hashes = hash_windows(seq1, window_len)

    N_matches=0
    N_windows=0
    for i in range(len(seq2) - window_len + 1):
        N_windows+=1
        window = seq2[i:i+window_len]
        if tuple(sorted(window)) in seq1_hashes:
            print('boom')
            N_matches+=1

    return N_matches/N_windows
                
                
    

#rastermap_pipeline()

    