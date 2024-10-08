import os
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

output_dir = '/home/maria/AllenData'
manifest_path = os.path.join(output_dir, "manifest.json")

functional_conn_session=[831882777]
brain_obs_session=[757970808]

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
func_session = cache.get_session_data(functional_conn_session[0])

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

print(set(df_one_more_repeats['frame'].values))