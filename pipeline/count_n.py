import torch
import torch.nn as nn
import torch.optim as optim
   
import os
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import numpy as np
import pickle

from lib_clean import get_spike_intervals

# Sample LNP Model definition
class LNPModel(nn.Module):
    def __init__(self, input_dim):
        super(LNPModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear layer for each neuron
    
    def forward(self, x):
        x = self.linear(x)
        firing_rate = torch.exp(x)  # Exponential non-linearity
        return firing_rate
    
output_dir = '/home/maria/AllenData'
manifest_path = os.path.join(output_dir, "manifest.json")

functional_conn_session=[831882777]
brain_obs_session=[757970808]

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
func_session = cache.get_session_data(brain_obs_session[0])

stimuli_df=func_session.stimulus_presentations
print(stimuli_df) 
print(stimuli_df.columns)

# Filter for the first stimulus of interest
df_one_more_repeats = stimuli_df[stimuli_df['stimulus_name'] == 'natural_scenes']
print(set(df_one_more_repeats['frame'].values))
print(len(df_one_more_repeats))
df_one_more_repeats = stimuli_df[stimuli_df['stimulus_name'] == 'natural_movie_one']
print(len(df_one_more_repeats))
df_one_more_repeats = stimuli_df[stimuli_df['stimulus_name'] == 'natural_movie_three']
print(len(df_one_more_repeats))