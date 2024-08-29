from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
import os
from dotenv import load_dotenv

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

def plot_cells(experiment_container_id, experiment_type):
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
    print(data_set_regression.get_dff_traces()[1].shape)
    print(data_set_events.shape)
    print(data_set_regression)
    print(cells)
    cell_indexes=range(0, len(cells))
    get_cell_trials_f(get_n_events, 0, data_set_events)


def get_cell_trials_f(action_function, cell_index, data_set):
    #timestamps, events = data_set.get_events(cell_specimen_id)
    #print(events)
    pass

def cell_compute_n_events(experiment_id, cell_specimen_id):
    data_set = boc.get_ophys_experiment_data(experiment_id)
    #timestamps, dff = data_set.get_dff_traces(cell_spec

plot_cells(experiment_container_id, 'three_session_C')
