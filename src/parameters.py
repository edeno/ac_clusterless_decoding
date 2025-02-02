from os.path import abspath, dirname, join, pardir

import numpy as np

from loren_frank_data_processing import Animal

# LFP sampling frequency
SAMPLING_FREQUENCY = 500

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')
FIGURE_DIR = join(ROOT_DIR, 'figures')

ANIMALS = {
    'jaq': Animal(directory='/stelmo/alison/jaq/filterframework', short_name='jaq'),
    'chimi': Animal(directory='/stelmo/alison/chimi/filterframework', short_name='chimi'),
}

EDGE_ORDER = [(3, 6),
              (6, 8),
              (6, 9),
              (3, 1),
              (1, 2),
              (1, 0),
              (3, 4),
              (4, 5),
              (4, 7)
             ]
EDGE_SPACING = 15

detector_parameters = {
    'movement_var': 6.0,
    'replay_speed': 1,
    'place_bin_size': 2.5,
    'spike_model_knot_spacing': 8.0,
    'spike_model_penalty': 0.5,
    'movement_state_transition_type': 'random_walk',
    'multiunit_model_kwargs': {
        'bandwidth': np.array([20.0, 20.0, 20.0, 20.0, 8.0])},
    'multiunit_occupancy_kwargs': {'bandwidth': np.array([8.0])},
    'discrete_state_transition_type': 'constant',
    'discrete_diagonal': np.array([0.00003, 0.968])
}

classifier_parameters = {
    'movement_var': 6.0,
    'replay_speed': 1,
    'place_bin_size': 2.5,
    'continuous_transition_types': [['random_walk', 'uniform'],
                                    ['uniform',     'uniform']],
    'model_kwargs': {
        'bandwidth': np.array([20.0, 20.0, 20.0, 20.0, 8.0])}

}

discrete_state_transition = np.array([[0.968, 0.032],
                                      [0.032, 0.968]])

'''
1. Geometric mean of duration is 1 / (1 - p)
2. So p = 1 - (1 / n_time_steps).
3. Want `n_time_steps` to equal half a theta cycle.
4. Theta cycles are ~8 Hz or 125 ms per cycle.
5. Half a theta cycle is 62.5 ms.
6. If our timestep is 2 ms, then n_time_steps = 31.25
7. So p = 0.968
'''
