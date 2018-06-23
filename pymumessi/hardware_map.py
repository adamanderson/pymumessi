# The entire hardware specification of an experiment is stored as a pandas
# DataFrame. It is straightforward to search and slice this object for
# subsets on which to run algorithms. This module contains a function
# for building the hardware map DataFrame with the minimal mandatory
# information contenct, as well as miscellaneous other functions for
# manipulating and checking hardware map DataFrames.

import pandas as pd
import numpy as np

mandatory_fields = ['board', 'channel_index', 'frequency', 'attenuation']

def build(resonator_files):
    '''
    Constructs a hardware map pandas DataFrame from a list of filenames. Files
    are assumed to be tab-separated text files containing the fields given in
    'mandatory_fields'.

    Parameters
    ----------
    resonator_files : list of str
        List of filenames containing resonator channel information

    Outputs
    -------
    hwm : pandas DataFrame
        Hardware map
    '''

    hwm = pd.DataFrame()
    for fname in resonator_files:
        board_hwm = pd.read_csv(fname, sep='\t')
        if any([field not in board_hwm.columns for field in mandatory_fields]):
            raise AttributeError('Mandatory field {} not in hardware map '
                                 'file {}.'.format(fname))
        hwm = hwm.append(board_hwm)

    return hwm
