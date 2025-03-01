import os
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import numpy as np

output_dir = '/home/maria/AllenData'
manifest_path = os.path.join(output_dir, "manifest.json")

functional_conn_session=[831882777]
brain_obs_session=[757970808]

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
func_session = cache.get_session_data(functional_conn_session[0])

#These stimuli have 900 frames per movie
stimuli_of_interest= ['natural_movie_one_more_repeats', 'natural_movie_one_shuffled']

stimuli_df=func_session.stimulus_presentations
print(stimuli_df) 
print(stimuli_df.columns)

# Filter for the first stimulus of interest
df_one_more_repeats = stimuli_df[stimuli_df['stimulus_name'] == 'natural_movie_one_more_repeats']

# Filter for the second stimulus of interest
df_one_shuffled = stimuli_df[stimuli_df['stimulus_name'] == 'natural_movie_one_shuffled']

print(len(df_one_more_repeats))
print(len(df_one_shuffled))

#print(set(df_one_more_repeats['frame'].values))
print(df_one_shuffled)

spike_times=func_session.spike_times

neurons=spike_times.keys()

print(df_one_more_repeats['start_time'].values, df_one_more_repeats['stop_time'].values)


def get_spikes_in_intervals_optimized(spike_times, start_times, stop_times):
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

# Example usage:
start_times = df_one_more_repeats['start_time'].values
stop_times = df_one_more_repeats['stop_time'].values
import time
start=time.time()
spikes_per_interval_optimized = get_spikes_in_intervals_optimized(spike_times, start_times, stop_times)
end=time.time()
print('Time taken: ', end-start)
print(spikes_per_interval_optimized)
print(spikes_per_interval_optimized[951109619].shape)
print(len(spikes_per_interval_optimized.keys()))
