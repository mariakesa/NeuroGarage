import numpy as np

def get_random_data_for_one_pair(lfp, spike_times, window_duration=0.25, bin_size=0.01, sampling_rate=1250.000067):
    """
    Selects a random time window from the LFP data and bins the spike times within that window for each unit.

    Parameters:
    - lfp (array-like): The LFP signal array with a 'time' attribute.
    - spike_times (list of numpy.ndarray): List containing spike time arrays for each unit.
    - window_duration (float): Duration of the time window in seconds (default: 0.25).
    - bin_size (float): Size of each bin in seconds for spike counting (default: 0.01).
    - sampling_rate (int): Sampling rate of the LFP in Hz (default: 1252).

    Returns:
    - lfp_in_window (numpy.ndarray): LFP data within the selected time window.
    - binned_spikes (numpy.ndarray): 2D array of spike counts per unit per bin within the window.
    """
    # Extract the time array from LFP data
    times = np.array(lfp.time)  # Assuming lfp.time is a numpy array of timestamps

    # Select a random start time ensuring the window fits within the LFP data
    random_time = np.random.uniform(low=times[0], high=times[-1] - window_duration)
    upper_bound_time = random_time + window_duration

    # Find the index corresponding to the random_time using searchsorted for efficiency
    lfp_ind = np.searchsorted(times, random_time)

    # Calculate the number of samples based on the sampling rate and window duration
    num_samples = int(window_duration * sampling_rate)  # e.g., 0.25 * 1252 ≈ 313 samples
    lfp_in_window = lfp[lfp_ind:lfp_ind + num_samples]  # Assuming lfp.signal contains the LFP data

    # Initialize a list to hold binned spike counts for each unit
    binned_spikes_list = []

    # Define bin edges for the spike histogram
    num_bins = int(window_duration / bin_size)
    bins = np.linspace(random_time, upper_bound_time, num_bins + 1)

    # Iterate over each unit's spike times and bin the spikes within the window
    for unit_spike_times in spike_times:
        # Extract spikes within the current window for this unit
        spikes_in_window = unit_spike_times[(unit_spike_times > random_time) & (unit_spike_times < upper_bound_time)]
        # Bin the spikes using numpy.histogram
        spike_counts, _ = np.histogram(spikes_in_window, bins=bins)
        
        # Append the spike counts to the list
        binned_spikes_list.append(spike_counts)
    
    # Convert the list of spike counts to a 2D numpy array (units x bins)
    binned_spikes = np.array(binned_spikes_list)  # Shape: (82, 25) for 82 units and 25 bins

    return lfp_in_window, binned_spikes
