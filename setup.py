#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy', 'numba', 'scipy', 'scikit-learn', 'matplotlib',
                    'xarray', 'pandas', 'dask', 'tqdm', 'statsmodels', 'patsy',
                    'spectral_connectivity', 'replay_trajectory_classification',
                    'ripple_detection', 'loren_frank_data_processing', 
                    'trajectory_analysis_tools', 'track_linearization']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='ac_clusterless_decoding',
    version='0.1.0.dev0',
    license='MIT',
    description=(''),
    author='',
    author_email='',
    url='',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
