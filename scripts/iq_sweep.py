from pymumessi.roach import Roach2
from pymumessi import hardware_map
import argparse as ap
import ConfigParser
import numpy as np

def load_frequencies(hwm):
    roaches = hardware_map.get_roaches(hwm)
    for roach in roaches:
        roach.load_freq(hwm)

        
def define_roach_lut(hwm):
    roaches = hardware_map.get_roaches(hwm)
    for roach in roaches:
        roach.defineRoachLUTs()


def define_dac_lut(hwm):
    roaches = hardware_map.get_roaches(hwm)
    for roach in roaches:
        roach.defineDacLUTs()
        



if __name__ == '__main__':
    P = ap.ArgumentParser(description='Initialize readout boards.',
                          formatter_class=ap.RawTextHelpFormatter)
    P.add_argument('configFile', action='store', help='Config file to load')
    P.add_argument('--LOstart', default=None, action='store', type=float,
                   help='Start frequency of LO sweep')
    P.add_argument('--LOstop', default=None, action='store', type=float,
                   help='Stop frequency of LO sweep')
    P.add_argument('--LOstep', default=None, action='store', type=float,
                   help='Frequency step of LO sweep')
    args = P.parse_args()

    hwm = hardware_map.build(args.configFile)

    # connect to the roach
    load_frequencies(hwm)
    define_roach_lut(hwm)
    define_dac_lut(hwm)

    # get some numbers needed for the sweep
    roaches = hardware_map.get_roaches(hwm)
    for roach in roaches:
        LO_freq = roach.LOFreq
        LO_span = roach.sweeplospan
        LO_step = roach.sweeplostep
        
        if args.LOstart:
            LO_start = args.LOstart
        else:
            LO_start = LO_freq - LO_span/2.

        if args.LOstop:
            LO_stop = args.LOstop
        else:
            LO_stop = LO_freq + LO_span/2.

        LO_offsets = np.arange(LO_start, LO_stop, LO_step) - LO_freq

        start_DACAtten = roach.dacatten_start
        stop_DACAtten = roach.dacatten_stop
        start_ADCAtten = roach.adcatten

        # run the sweep itself
        sweep_data = roach.performIQSweep(LO_start/1.e6, LO_stop/1.e6, LO_step/1.e6)
