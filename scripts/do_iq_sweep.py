from pymumessi.roach import Roach2
import argparse as ap
import ConfigParser
import numpy as np

P = ap.ArgumentParser(description='Initialize readout boards.',
                                            formatter_class=ap.RawTextHelpFormatter)

P.add_argument('roachNum', default=100, action='store', type=int,
               help='Roach number to initialize')
P.add_argument('--LOstart', default=None, action='store', type=float,
               help='Start frequency of LO sweep')
P.add_argument('--LOstop', default=None, action='store', type=float,
               help='Stop frequency of LO sweep')
P.add_argument('--LOstep', default=None, action='store', type=float,
               help='Frequency step of LO sweep')
P.add_argument('-c', '--config', action='store', default='init.cfg',
               help='Config file to load')
args = P.parse_args()

# parse config file
config = ConfigParser.ConfigParser()
config.read(args.config)

# connect to the roach
roach_board = Roach2(args.roachNum, args.config)
roach_board.loadFreq()
roach_board.defineRoachLUTs()
roach_board.defineDacLUTs()

# get some numbers needed for the sweep
LO_freq = roach_board.LOFreq
LO_span = config.getfloat('Roach '+str(args.roachNum),'sweeplospan')

if args.LOstart:
    LO_start = args.LOstart
else:
    LO_start = LO_freq - LO_span/2.

if args.LOstop:
    LO_stop = args.LOstop
else:
    LO_stop = LO_freq + LO_span/2.

LO_step = config.getfloat('Roach '+str(args.roachNum),'sweeplostep')
LO_offsets = np.arange(LO_start, LO_stop, LO_step) - LO_freq

start_DACAtten = config.getfloat('Roach '+str(args.roachNum),'dacatten_start')
stop_DACAtten = config.getfloat('Roach '+str(args.roachNum),'dacatten_stop')
start_ADCAtten = config.getfloat('Roach '+str(args.roachNum),'adcatten')

# run the sweep itself
sweep_data = roach_board.performIQSweep(LO_start, LO_stop, LO_step)
