import os
from logging import getLogger

import numpy as np
import pandas as pd
from loren_frank_data_processing import (get_all_multiunit_indicators,
                                         make_tetrode_dataframe)
from loren_frank_data_processing.core import (get_data_structure,
                                              reconstruct_time)
from loren_frank_data_processing.position import _get_pos_dataframe
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter1d
from src.parameters import (ANIMALS, EDGE_ORDER, EDGE_SPACING,
                            SAMPLING_FREQUENCY)
from track_linearization import get_linearized_position, make_track_graph

logger = getLogger(__name__)


def gaussian_smooth(data, sigma, sampling_frequency, axis=0, truncate=8):
    '''1D convolution of the data with a Gaussian.

    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is just a wrapper around scipy's
    `gaussian_filter1d`, The support is truncated at 8 by default, instead
    of 4 in `gaussian_filter1d`

    Parameters
    ----------
    data : array_like
    sigma : float
    sampling_frequency : int
    axis : int, optional
    truncate : int, optional

    Returns
    -------
    smoothed_data : array_like

    '''
    return gaussian_filter1d(
        data, sigma * sampling_frequency, truncate=truncate, axis=axis,
        mode='constant')


def get_multiunit_population_firing_rate(multiunit, sampling_frequency,
                                         smoothing_sigma=0.015):
    '''Calculates the multiunit population firing rate.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_signals)
        Binary array of multiunit spike times.
    sampling_frequency : float
        Number of samples per second.
    smoothing_sigma : float or np.timedelta
        Amount to smooth the firing rate over time. The default is
        given assuming time is in units of seconds.


    Returns
    -------
    multiunit_population_firing_rate : ndarray, shape (n_time,)

    '''
    return gaussian_smooth(multiunit.mean(axis=1) * sampling_frequency,
                           smoothing_sigma, sampling_frequency)


def _get_pos_dataframe(epoch_key, animals):
    animal, day, epoch = epoch_key
    struct = get_data_structure(animals[animal], day, 'pos', 'pos')[epoch - 1]
    position_data = struct['data'][0, 0]
    FIELD_NAMES = ['time', 'x_position', 'y_position', 'head_direction',
                   'speed', 'smoothed_x_position', 'smoothed_y_position',
                   'smoothed_head_direction', 'smoothed_speed']
    time = pd.TimedeltaIndex(
        position_data[:, 0], unit='s', name='time')
    n_cols = position_data.shape[1]

    if n_cols > 5:
        # Use the smoothed data if available
        NEW_NAMES = {'smoothed_x_position': 'x_position',
                     'smoothed_y_position': 'y_position',
                     'smoothed_head_direction': 'head_direction',
                     'smoothed_speed': 'speed'}
        return (pd.DataFrame(
            position_data[:, 5:9], columns=FIELD_NAMES[5:9], index=time)
            .rename(columns=NEW_NAMES))
    else:
        return pd.DataFrame(position_data[:, 1:5], columns=FIELD_NAMES[1:5],
                            index=time)


def get_interpolated_position_info(
        epoch_key, animals, use_HMM=False,
        route_euclidean_distance_scaling=1.0, sensor_std_dev=5.0,
        diagonal_bias=0.5, edge_spacing=EDGE_SPACING,
        edge_order=EDGE_ORDER):

    position_info = _get_pos_dataframe(epoch_key, animals)
    position_info = position_info.resample('2ms').mean().interpolate('time')
    position_info.loc[
        position_info.speed < 0, 'speed'] = 0.0
    track_graph = get_track_graph()
    position = np.asarray(position_info.loc[:, ['x_position', 'y_position']])

    linear_position_df = get_linearized_position(
        position=position,
        track_graph=track_graph,
        edge_spacing=EDGE_SPACING,
        edge_order=EDGE_ORDER,
        use_HMM=use_HMM,
        route_euclidean_distance_scaling=route_euclidean_distance_scaling,
        sensor_std_dev=sensor_std_dev,
        diagonal_bias=diagonal_bias,
    ).set_index(position_info.index)

    position_info = pd.concat((position_info, linear_position_df), axis=1)

    return position_info


def get_track_graph():
    NODE_POSITIONS = np.array([
        (79.910, 216.720),  # top left well 0
        (132.031, 187.806),  # top middle intersection 1
        (183.718, 217.713),  # top right well 2
        (132.544, 132.158),  # middle intersection 3
        (87.202, 101.397),  # bottom left intersection 4
        (31.340, 126.110),  # middle left well 5
        (180.337, 104.799),  # middle right intersection 6
        (92.693, 42.345),  # bottom left well 7
        (183.784, 45.375),  # bottom right well 8
        (231.338, 136.281),  # middle right well 9
    ])

    EDGES = np.array([
        (0, 1),
        (1, 2),
        (1, 3),
        (3, 4),
        (4, 5),
        (3, 6),
        (6, 9),
        (4, 7),
        (6, 8),
    ])

    return make_track_graph(NODE_POSITIONS, EDGES)


def load_data(epoch_key):
    logger.info('Loading position information and linearizing...')
    position_info = (get_interpolated_position_info(epoch_key, ANIMALS)
                     .dropna(subset=["linear_position"]))

    logger.info('Loading multiunits...')
    tetrode_info = make_tetrode_dataframe(
        ANIMALS).xs(epoch_key, drop_level=False)
    tetrode_keys = tetrode_info.loc[tetrode_info.area == 'hpc'].index
    # ca1 for jaq

    def _time_function(*args, **kwargs):
        return position_info.index

    multiunits = get_all_multiunit_indicators(
        tetrode_keys, ANIMALS, _time_function)

    multiunit_spikes = (np.any(~np.isnan(multiunits.values), axis=1)
                        ).astype(np.float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(
            multiunit_spikes, SAMPLING_FREQUENCY), index=position_info.index,
        columns=['firing_rate'])

    logger.info('Loading theta...')
    is_ref = (tetrode_info.reset_index()
              .tetrode_number.isin(tetrode_info.ref.dropna().unique())).values
    ref_tetrode_key = tetrode_info.loc[is_ref].index[0]
    theta_df = get_filter(ref_tetrode_key, ANIMALS, freq_band='theta')
    track_graph = get_track_graph()

    return {
        'position_info': position_info,
        'multiunits': multiunits,
        'theta': theta_df,
        'multiunit_firing_rate': multiunit_firing_rate,
        'track_graph': track_graph,
    }


def get_filter_filename(tetrode_key, animals, freq_band='theta'):
    '''Returns a file name for the filtered LFP for an epoch.

    Parameters
    ----------
    tetrode_key : tuple
        Unique key identifying the tetrode. Elements are
        (animal_short_name, day, epoch, tetrode_number).
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    filename : str
        File path to tetrode file LFP
    '''
    animal, day, epoch, tetrode_number = tetrode_key
    # add eeggnd to filename to get correct theta from a reference ntrode
    filename = (f'{animals[animal].short_name}{freq_band}eeggnd{day:02d}'
                f'-{epoch}-{tetrode_number:02d}.mat')
    return os.path.join(animals[animal].directory, 'EEG', filename)


def get_filter(tetrode_key, animals, freq_band='theta'):
    filter_file = loadmat(
        get_filter_filename(tetrode_key, animals, freq_band))
    filter_data = filter_file[freq_band][0, -1][0, -1][0, -1][0]
    time = reconstruct_time(
        filter_data['starttime'][0][0][0],
        filter_data['data'][0].shape[0],
        float(filter_data['samprate'][0][0][0]))

    COLUMNS = ['bandpassed_lfp', 'instantaneous_phase', 'envelope_magnitude']
    df = pd.DataFrame(filter_data['data'][0], columns=COLUMNS, index=time)

    return df
