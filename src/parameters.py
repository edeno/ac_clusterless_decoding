from os.path import abspath, dirname, join, pardir

from loren_frank_data_processing import Animal

# LFP sampling frequency
SAMPLING_FREQUENCY = 500

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')
FIGURE_DIR = join(ROOT_DIR, 'figures')

ANIMALS = {
    'jaq': Animal(directory=join(RAW_DATA_DIR, 'jaq'), short_name='jaq'),
}

EDGE_ORDER = [6, 5, 3, 8, 7, 4, 2, 0, 1]
EDGE_SPACING = 15
