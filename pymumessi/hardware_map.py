# The entire hardware specification of an experiment is stored as a pandas
# DataFrame. It is straightforward to search and slice this object for
# subsets on which to run algorithms. The DataFrame specification contains
# one row per channel, and each column specifies which hardware is connected
# to that channel upstream.
#
# DataFrame entries are all basic Python types (strings, ints, and floats),
# with one exception. It is useful to be able to store board state information
# in roach objects. To allow this information to persist and to avoid having
# to separately pass the hardware map and the matching collection of Roach2
# objects, one Roach2 object for each board is initialized and stored as an
# additional column in the hardware map. The Roach2 objects are initialized in
# the 'build' function.

import pandas as pd
import numpy as np
from pymumessi.roach import Roach2
import ConfigParser

mandatory_fields = ['board', 'channel_index', 'frequency', 'attenuation']
field_dtypes = {'board': int,
                'channel_index': int,
                'frequency': float,
                'attenuation': float}
def build(config_file, resonator_files=None):
    '''
    Constructs a hardware map pandas DataFrame from a list of filenames. Files
    are assumed to be tab-separated text files containing the fields given in
    'mandatory_fields'.

    Parameters
    ----------
    config_file : str
        Path to pymumessi config file containing hwm filenames for each board
    resonator_files : list of str
        Optional list of filenames containing resonator channel information;
        overrides default behavior of using the hwm filenames specified in the
        master config file

    Outputs
    -------
    hwm : pandas DataFrame
        Hardware map
    '''

    hwm = pd.DataFrame()

    if resonator_files is None:
        CP = ConfigParser.ConfigParser()
        CP.read(config_file)
        resonator_files = [CP.get(section, 'hwm') for section in CP.sections()
                           if 'Roach' in section]

    for fname in resonator_files:
        board_hwm = pd.read_csv(fname, sep='\t', dtype=field_dtypes)
        if any([field not in board_hwm.columns for field in mandatory_fields]):
            raise AttributeError('Mandatory field {} not in hardware map '
                                 'file {}.'.format(fname))
        hwm = hwm.append(board_hwm)

    # create board objects in an additional column
    boards = {board: Roach2(board, config_file)
              for board in np.unique(hwm['board'])}
    hwm['Roach2_object'] = pd.Series([boards[boardnum]
                                   for boardnum in hwm['board']],
                                   index=hwm.index)

    validate_hwm(CP, hwm)
    
    return hwm


def validate_hwm(config, hwm):
    '''
    Check the validity of a hwm and config file.

    Parameters
    ----------
    config : ConfigParser
        ConfigParser object corresponding to pymumessi config file 
    hwm : pandas DataFrame
        Hardware map data frame

    Returns
    -------
    None
    '''
    # check whether all boards in config file are represented in hwm and
    # vice-versa
    boards_in_config = [int(section.lstrip('Roach '))
                        for section in config.sections() if 'Roach' in section]
    boards_in_hwm = np.unique(hwm['board'])
    if np.array_equal(boards_in_config, boards_in_hwm) is False:
        raise KeyError('Board numbers present in config file do not match '
                       'board numbers present in hardware map files.')
